import torch
import torch.nn as nn
import torch.nn.functional as F
#import openai
#import graph_prompt as Prompt
import matplotlib.pyplot as plt
from torch_geometric.nn.inits import glorot
from torch import Tensor
from fvcore.nn import FlopCountAnalysis

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dassl.utils import load_pretrained_weights, load_checkpoint
import math
from dassl.metrics import compute_accuracy

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

NEGATIVE_TEMPLATES = {
    "OxfordPets": "not a photo of a {}, not a type of pet, no distortions, no artifacts, no watermarks.",
    "OxfordFlowers": "not a photo of a {}, not a type of flower, avoid blurry or oversaturated images, no watermarks.",
    "FGVCAircraft": "not a photo of a {}, not a type of aircraft, avoid deformations, low quality or artifacts.",
    "DescribableTextures": "not a {} texture, no repetitive patterns, no blurring or artifacts.",
    "EuroSAT": "not a centered satellite photo of {}, avoid low resolution and artifacts.",
    "StanfordCars": "not a photo of a {}, not a type of car, no distortions, no extra parts, no artifacts.",
    "Food101": "not a photo of {}, not a type of food, avoid low quality, extra text, or watermarks.",
    "SUN397": "not a photo of a {}, avoid unnatural colors, distortions, or watermarks.",
    "Caltech101": "not a photo of a {}, no distortions, artifacts, or extra markings.",
    "UCF101": "not a photo of a person doing {}, avoid motion blur, distortions, and extra limbs.",
    "ImageNet": "not a photo of a {}, avoid common image generation artifacts, blurriness, or distortions.",
    "ImageNetSketch": "not a sketch of a {}, avoid unclear outlines, deformations, or extra marks.",
    "ImageNetV2": "not a photo of a {}, avoid low quality, distortions, or any additional unwanted artifacts.",
    "ImageNetA": "not a photo of a {}, no unwanted details, distortions, or extra artifacts.",
    "ImageNetR": "not a photo of a {}, avoid distortions, deformations, or extra irrelevant details.",
}

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

    model = clip.build_model(state_dict or model.state_dict())

    return model


def graph_norm_ours(A, batch=False, self_loop=True, symmetric=True):
    # A = A + I    A: (bs, num_nodes, num_nodes
    # Degree
    d = A.sum(-1)  # (bs, num_nodes) #[1000, m+1]
    if symmetric:
        # D = D^-1/2
        d = torch.pow(d, -0.5)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A).bmm(D)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A).mm(D)
    else:
        # D=D^-1
        d = torch.pow(d, -1)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A)

    return norm_A


def cal_similarity(x, p=2, dim=1):
    '''
    x: (n,K)
    return: (n,n)
    '''
    x = F.normalize(x, p=p, dim=dim)
    return torch.mm(x, x.transpose(0, 1))


def cal_edge_emb(x, p=2, dim=1):  # v1_graph---taking the similairty by

    x = F.normalize(x, p=p, dim=dim)
    x_c = x
    x = x.transpose(1, 2)
    x_r = x
    A = torch.bmm(x_r, x_c) 
    return A

class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        return x + self.global_emb


class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = torch.matmul(weight, self.p_list)

        return x + p

class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, name=None, device=None, class_num=None, sparse_inputs=False, act=nn.Tanh, bias=True,
                 dropout=0.0):
        super().__init__()
        self.act = nn.Tanh()
        self.device = device
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.hidden_dim = 512
        self.class_num = class_num
        self.gcn_weights = nn.Parameter(torch.ones(self.hidden_dim, self.hidden_dim))
        if self.bias:
            self.gcn_bias = nn.Parameter(torch.zeros(class_num, self.hidden_dim))

        # else:
        #     self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.gcn_weights.size(1))
        self.gcn_weights.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        x = feat  
        node_size = adj.size()[1]
        adj = torch.clip(adj, min=0.0)
        I = torch.eye(node_size, device='cuda').unsqueeze(dim=0).to(self.device)
        adj = adj + I  
        adj = graph_norm_ours(adj, batch=True, self_loop=True, symmetric=True)  #
        x = x.transpose(1, 2)
        pre_sup = torch.matmul(x, self.gcn_weights)  
        output = torch.matmul(adj, pre_sup)  

        if self.bias:
            output += self.gcn_bias.unsqueeze(1)
        if self.act is not None:
            return self.act(output[:, 0, :])
        else:
            return output[:, 0, :]


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class GraphLearner(nn.Module):
     def __init__(self, cfg, classnames, clip_model, base_text_features, base_img_features, base_Negative_text_features):
          super().__init__()
          self.device = clip_model.dtype
          self.alpha = 0.1
          print(">> DCT scale factor: ", self.alpha)
          self.register_buffer("base_text_features", base_text_features)  #
          self.register_buffer("base_img_features", base_img_features)
          self.register_buffer("base_Negative_text_features", base_Negative_text_features)
          # self.alpha_it = cfg.TRAINER.GRAPHADAPTER.ALPHA
          self.alpha_it = 0.7
          self.beta_it = 0.5
          self.node_num = 1
          # self.alpha_it =
          self.hidden_dim = 1

          self.prompt_text_learner = GPFplusAtt(base_text_features.size(1), base_text_features.size(0))
          self.prompt_image_learner = GPFplusAtt(base_img_features.size(1), base_img_features.size(0))

          self.GCN_tt = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device, class_num=base_text_features.size()[0])

          self.GCN_it = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device, class_num=base_text_features.size()[0])

          self.GCN_ntt = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device, class_num=base_text_features.size()[0])


     def reset_parameters(self):
          for i in range(self.node_num):
               stdv = 1. / math.sqrt(self.graph_node[i].size(0))
               self.graph_node[i].data.uniform_(-stdv, stdv)

     def forward(self, img_feature):

          with torch.no_grad():
               node_cluster_t = self.base_text_features.view(1, self.base_text_features.size()[0], self.base_text_features.size()[1])

               node_cluster_i = self.base_img_features.view(1, self.base_img_features.size()[0], self.base_img_features.size()[1])

               node_cluster_nt = self.base_Negative_text_features.view(1, self.base_Negative_text_features.size()[0], self.base_Negative_text_features.size()[1])
          graph_o_t_all = []
          for index in range(4):
               with torch.no_grad():
                    inputs_text = self.base_text_features.unsqueeze(dim=1)
                    node_cluster_tt = node_cluster_t[:, :, :].repeat(inputs_text.size()[0], 1, 1)
                    node_cluster_it = node_cluster_i[:, :, :].repeat(inputs_text.size()[0], 1, 1)
                    node_cluster_ntt = node_cluster_nt[:, :, :].repeat(inputs_text.size()[0], 1, 1)
                    feat_tt = torch.cat([inputs_text, node_cluster_tt], dim=1)
                    feat_it = torch.cat([inputs_text, node_cluster_it], dim=1)
                    feat_ntt = torch.cat([inputs_text, node_cluster_ntt], dim=1)

                    feat_tt = feat_tt.transpose(1, 2)
                    feat_it = feat_it.transpose(1, 2)
                    feat_ntt = feat_ntt.transpose(1, 2)
                    edge_tt = cal_edge_emb(feat_tt).detach()
                    edge_it = cal_edge_emb(feat_it).detach()
                    edge_ntt = cal_edge_emb(feat_ntt).detach()

               feat_tt = feat_tt.transpose(1, 2)
               feat_it = feat_it.transpose(1, 2)
               feat_tt = self.prompt_text_learner.add(feat_tt)
               feat_it = self.prompt_image_learner.add(feat_it)  
               feat_tt = feat_tt.transpose(1, 2)
               feat_it = feat_it.transpose(1, 2)
               graph_o_tt = self.GCN_tt(feat_tt, edge_tt)
               graph_o_it = self.GCN_it(feat_it, edge_it)
               graph_o_ntt = self.GCN_ntt(feat_ntt, edge_ntt)
               graph_o_t = (graph_o_tt) * self.alpha_it + (1 - self.alpha_it) * graph_o_it
               graph_o_t_all.append(graph_o_t)
          graph_o_t = torch.stack(graph_o_t_all, dim=0).mean(dim=0)

          return self.beta_it * self.base_text_features + (1 - self.beta_it) * graph_o_t.squeeze(), graph_o_tt, graph_o_it, graph_o_ntt, img_feature


def _get_base_image_features(cfg, classnames, clip_model, img_encoder, train_loader_x):
    device = next(img_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        img_encoder = img_encoder.cuda()
    with torch.no_grad():
        img_feature = []
        labels = []
        for epch in range(10):
            for batch_idx, batch in enumerate(train_loader_x):
                image = batch["img"]
                label = batch["label"]
                image = image.cuda()
                label = label.cuda()
                image_features = img_encoder(image.type(clip_model.dtype)).detach()
                img_feature.append(image_features)
                labels.append(label)
        img_feature_list = torch.cat(img_feature, dim=0)
        label_list = torch.cat(labels, dim=0)
        unique_labels = torch.unique(label_list)

        img_feature_list_all = []

        for label in unique_labels:
            class_features = img_feature_list[label_list == label]
            class_mean_feature = class_features.mean(dim=0)
            img_feature_list_all.append(class_mean_feature.unsqueeze(0))
        img_feature_list_all = torch.cat(img_feature_list_all, dim=0)
        img_encoder = img_encoder.to(device)

    return img_feature_list_all.to(device)


def _get_base_text_features(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()
    dataset = cfg.DATASET.NAME

    if dataset == "ImageNet":
        TEMPLATES = IMAGENET_TEMPLATES_SELECT
    else:
        TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in TEMPLATES])
            tokens = tokens.to(device)
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))  
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)

def _get_negative_text_features(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()
    dataset = cfg.DATASET.NAME

    if dataset == "ImageNet":
        Negative_TEMPLATES = IMAGENET_TEMPLATES_SELECT
    else:
        Negative_TEMPLATES = []

    Negative_TEMPLATES += [NEGATIVE_TEMPLATES[dataset]]

    with torch.no_grad():
        Negative_text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in Negative_TEMPLATES])
            tokens = tokens.to(device)
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                Negative_text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))  # not support float16 on cpu
            else:
                Negative_text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    Negative_text_embeddings = torch.stack(Negative_text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return Negative_text_embeddings.to(device)

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, train_loader_x):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype  # float16
        text_encoder = TextEncoder(clip_model)
        img_encoder = self.image_encoder
        base_text_features = _get_base_text_features(cfg, classnames, clip_model, text_encoder)
        base_img_features = _get_base_image_features(cfg, classnames, clip_model, img_encoder, train_loader_x)
        base_Negative_text_features = _get_negative_text_features(cfg, classnames, clip_model, text_encoder)

        self.graph_learner = GraphLearner(cfg, classnames, clip_model, base_text_features, base_img_features, base_Negative_text_features)

    def forward(self, image, mode='train'):
        if not self.training:
            mode = 'test'
        try:
            image_features = self.image_encoder(image.type(self.dtype)).detach()
        except:
            image_features = self.image_encoder(image.float()).detach()

        text_features, graph_o_tt, graph_o_it, graph_o_ntt, image_features = self.graph_learner(image_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if mode == 'train':
            return logits, graph_o_tt, graph_o_it, graph_o_ntt
        else:
            if isinstance(logits, tuple):
                return logits[0]
            return logits


@TRAINER_REGISTRY.register()
class GraphCLIP_v1(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.train_loader_x).cuda()
        for name, param in self.model.named_parameters():
            if "graph_learner" not in name:
                param.requires_grad_(False)

        for param in self.model.graph_learner.parameters():
            param.requires_grad_(True)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.graph_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.float()
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(model=self.model.graph_learner, optim_cfg=cfg.OPTIM)
        #    , optim_cfg, param_groups=None
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("graph_learner", self.model.graph_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None



    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, graph_o_tt, graph_o_it, graph_o_ntt = self.model(image, mode='train')
            ce_loss = F.cross_entropy(output, label)
            triplet_loss = F.triplet_margin_loss(
                graph_o_it,
                graph_o_tt,
                graph_o_ntt,
                margin=1.0, 
                p=2
            )

            total_loss = ce_loss + triplet_loss
            self.model_backward_and_update(total_loss)

        loss_summary = {
            "loss": total_loss.item(),
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


