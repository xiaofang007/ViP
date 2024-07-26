import os.path as osp
import torch
import torch.nn as nn
import json
from collections import OrderedDict
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'ViP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class MergePrompt(nn.Module):
    def __init__(self,dim,dtype):
        super().__init__()
        self.dim = dim
        self.scale = dim**(-0.5)
        self.q_proj = nn.Linear(dim, dim, bias=False,dtype=dtype)
        self.k_proj = nn.Linear(dim, dim, bias=False,dtype=dtype)
        nn.init.xavier_normal_(self.k_proj.weight)
        nn.init.xavier_normal_(self.q_proj.weight)

    def forward(self, query, key=None, value=None):
        if key is None:
            key = query
        if value is None:
            value = key

        q = self.q_proj(query)
        k = self.k_proj(key)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = (query @ key.transpose(-2, -1)) * self.scale

        attn = torch.softmax(attn,dim=-1)

        # out = attn @ v
        out = attn @ value

        out = query + out
        return out


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final   # layernorm
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        out = OrderedDict()
        for i,(k,v) in enumerate(prompts.items()):
            x = v + self.positional_embedding.type(self.dtype)
            x  = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), tokenized_prompts[k].argmax(dim=-1)]
            x = x @ self.text_projection  # text_projection: (512,1024)
            out[k] = x

        return out


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)  # num class = 100
        n_ctx = cfg.TRAINER.VIP.N_CTX  # num context token = 16
        # ctx_init = cfg.TRAINER.VIP.CTX_INIT   
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]   # text feature shape
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        with open(cfg.DESCRIPTOR_PATH, 'r') as fp:
            gpt_descriptions = json.load(fp)

        # random initialization
        if cfg.TRAINER.VIP.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # learnable token para to be optimized

        tokenized_prompts = OrderedDict()
        embedding = OrderedDict()
        name_lens = OrderedDict()
        for i, (k, v) in enumerate(gpt_descriptions.items()):
            name_lens[k] = [len(_tokenizer.encode(item)) for item in v]
            prompts = [prompt_prefix + " " + item + "." for item in v]  
            tokenized_prompts[k] = torch.cat([clip.tokenize(p) for p in prompts]) 
            with torch.no_grad():
                embedding[k] = clip_model.token_embedding(tokenized_prompts[k]).type(dtype) 

        self.token_prefix = OrderedDict()
        self.token_suffix = OrderedDict()
        for i, (k, v) in enumerate(embedding.items()):
            self.token_prefix[k] = embedding[k][:, :1, :].cuda()
            self.token_suffix[k] = embedding[k][:, 1+n_ctx:, :].cuda()

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.VIP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        prompts = OrderedDict()
        for index, (k, v) in enumerate(self.tokenized_prompts.items()):
            prefix = self.token_prefix[k]
            suffix = self.token_suffix[k]

            if ctx.dim() == 2:
                ctx_k = ctx.unsqueeze(0).expand(prefix.shape[0], -1, -1)  # all prompt has the same template
            else:
                ctx_k = ctx[index].unsqueeze(0).expand(prefix.shape[0], -1, -1)  # class-specific prompt

            if self.class_token_position == "end":
                prompts[k] = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_k,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )

            elif self.class_token_position == "middle":
                single_desc_prompts = []
                half_n_ctx = self.n_ctx // 2
                for i in range(len(self.name_lens[k])):  # for class k, there are m descriptors
                    name_len = self.name_lens[k][i]
                    prefix_i = prefix[i:i+1,:,:]
                    descriptor_i = suffix[i:i+1, :name_len, :]
                    suffix_wo_desc_i = suffix[i:i+1, name_len:, :]
                    ctx_half_i_1 = ctx_k[i:i+1, :half_n_ctx, :]
                    ctx_half_i_2 = ctx_k[i:i+1, half_n_ctx:, :]
                    prompt = torch.cat(
                        [
                            prefix_i,     # (1, 1, dim)
                            ctx_half_i_1,  # (1, n_ctx//2, dim)
                            descriptor_i,      # (1, name_len, dim)
                            ctx_half_i_2,  # (1, n_ctx//2, dim)
                            suffix_wo_desc_i,     # (1, *, dim)
                        ],
                        dim=1,
                    )
                    single_desc_prompts.append(prompt)
                prompts[k] = torch.cat(single_desc_prompts, dim=0)

            elif self.class_token_position == "front":
                single_desc_prompts = []
                for i in range(len(self.name_lens[k])):
                    name_len = self.name_lens[k][i]
                    prefix_i = prefix[i : i + 1, :, :]
                    descriptor_i = suffix[i : i + 1, :name_len, :]
                    suffix_wo_desc_i = suffix[i : i + 1, name_len:, :]
                    ctx_k_i = ctx_k[i : i + 1, :, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            descriptor_i,   # (1, name_len, dim)
                            ctx_k_i,     # (1, n_ctx, dim)
                            suffix_wo_desc_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                    single_desc_prompts.append(prompt)
                prompts[k] = torch.cat(prompts, dim=0)

            else:
                raise ValueError

        return prompts  


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.classnames = classnames
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        token = torch.empty((len(classnames),clip_model.ln_final.weight.shape[0]),dtype=self.dtype)
        self.learnable_token = nn.Parameter(token) # learnable token
        nn.init.normal_(self.learnable_token, std=0.02)

        self.attn = MergePrompt(dim=clip_model.ln_final.weight.shape[0],dtype=self.dtype)

    def forward(self, image):
        B,C,H,W = image.shape
        logit_scale = self.logit_scale.exp()
        logits = torch.zeros((B,len(self.classnames)),dtype=self.dtype,requires_grad=True).cuda()
        image_features = self.image_encoder(image.type(self.dtype))   

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts) 

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        for i, (k, v) in enumerate(text_features.items()):
            grouped_text_features = self.attn(self.learnable_token[i:i+1,:],v)
            normalized_text_features = grouped_text_features/grouped_text_features.norm(dim=-1, keepdim=True)
            score = image_features @ normalized_text_features.t()
            logits[:,i:i+1] += logit_scale*score


        return logits


@TRAINER_REGISTRY.register()
class ViP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.VIP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.VIP.PREC == "fp32" or cfg.TRAINER.VIP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "learnable_token" not in name and "attn" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        opt_para = list(self.model.prompt_learner.parameters())+list(self.model.attn.parameters())
        opt_para.append(self.model.learnable_token)
        self.optim = build_optimizer(opt_para, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        module_state_dict = nn.ModuleDict()
        param_state_list = nn.ParameterList()
        param_state_list.append(self.model.learnable_token)
        module_state_dict['prompt_learner'] = self.model.prompt_learner
        module_state_dict['attn'] = self.model.attn
        module_state_dict['token'] = param_state_list
        self.register_model("vip", module_state_dict, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.VIP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.VIP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
