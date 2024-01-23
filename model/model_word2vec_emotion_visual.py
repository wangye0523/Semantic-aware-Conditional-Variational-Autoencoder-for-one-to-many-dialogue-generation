import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import numpy as np
from model.Embedding import Embedding
from model.Encoder import Encoder
from model.PriorNet import PriorNet
from model.RecognizeNet import RecognizeNet
from model.Decoder import Decoder
from model.PrepareState import PrepareState
import torch.nn.functional as F

emotion_num = 3


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        # 定义嵌入层
        self.embedding = Embedding(config.num_vocab,  # 词汇表大小
                                   config.embedding_size,  # 嵌入层维度
                                   config.pad_id,  # pad_id
                                   config.dropout)

        # post编码器
        self.post_encoder = Encoder(config.post_encoder_cell_type,  # rnn类型
                                    config.embedding_size,  # 输入维度
                                    config.post_encoder_output_size,  # 输出维度
                                    config.post_encoder_num_layers,  # rnn层数
                                    config.post_encoder_bidirectional,  # 是否双向
                                    config.dropout) # dropout概率

        # response编码器
        self.response_encoder = Encoder(config.response_encoder_cell_type,
                                        config.embedding_size,  # 输入维度
                                        config.response_encoder_output_size,  # 输出维度
                                        config.response_encoder_num_layers,  # rnn层数
                                        config.response_encoder_bidirectional,  # 是否双向
                                        config.dropout)  # dropout概率

        # 先验网络
        self.prior_net = PriorNet(config.post_encoder_output_size,  # post输入维度
                                  config.latent_size,  # 潜变量维度
                                  config.dims_prior)  # 隐藏层维度

        # 识别网络
        self.recognize_net = RecognizeNet(config.post_encoder_output_size,  # post输入维度
                                          config.response_encoder_output_size,  # response输入维度
                                          config.latent_size,  # 潜变量维度
                                          config.dims_recognize)  # 隐藏层维度

        # 初始化解码器状态
        self.prepare_state = PrepareState(config.post_encoder_output_size+config.latent_size,
                                          config.decoder_cell_type,
                                          config.decoder_output_size,
                                          config.decoder_num_layers)

        # 解码器
        self.decoder = Decoder(config.decoder_cell_type,  # rnn类型
                               config.embedding_size,  # 输入维度
                               config.decoder_output_size,  # 输出维度
                               config.decoder_num_layers,  # rnn层数
                               config.dropout)  # dropout概率
        #no_emotion decoder
        self.no_emotion_decoder = Decoder(config.decoder_cell_type,  # rnn类型
                               config.embedding_size,  # 输入维度
                               config.decoder_output_size,  # 输出维度
                               config.decoder_num_layers,  # rnn层数
                               config.dropout)  # dropout概率
        #positive decoder
        self.positive_decoder = Decoder(config.decoder_cell_type,  # rnn类型
                               config.embedding_size,  # 输入维度
                               config.decoder_output_size,  # 输出维度
                               config.decoder_num_layers,  # rnn层数
                               config.dropout)  # dropout概率
        #negative decoder
        self.negative_decoder = Decoder(config.decoder_cell_type,  # rnn类型
                               config.embedding_size,  # 输入维度
                               config.decoder_output_size,  # 输出维度
                               config.decoder_num_layers,  # rnn层数
                               config.dropout)  # dropout概率

        # 初始化解码器状态
        self.no_emotion_prepare_state = PrepareState(config.post_encoder_output_size + config.latent_size,
                                          config.decoder_cell_type,
                                          config.decoder_output_size,
                                          config.decoder_num_layers)

        # 初始化解码器状态
        self.positive_prepare_state = PrepareState(config.post_encoder_output_size + config.latent_size,
                                          config.decoder_cell_type,
                                          config.decoder_output_size,
                                          config.decoder_num_layers)

        # 初始化解码器状态
        self.negative_prepare_state = PrepareState(config.post_encoder_output_size + config.latent_size,
                                          config.decoder_cell_type,
                                          config.decoder_output_size,
                                          config.decoder_num_layers)

        #classification
        self.classification = nn.Sequential(
            nn.Linear(config.latent_size, emotion_num),
            nn.Softmax(-1)
        )

        # 输出层
        self.projector = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )

        self.projector_no_emotion = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )

        self.projector_positive = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )

        self.projector_negative = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )


    def forward(self, inputs,word2vec, inference=False, inpre=False, ingau=False, max_len=60, gpu=True):
        z_s = []
        label_s = []

        with torch.no_grad():
            for data in inputs.get_batch_data():
                inputs2 = self.prepare_feed_data(data)
                id_posts = inputs2['posts']  # [batch, seq]
                len_posts = inputs2['len_posts']  # [batch]
                id_responses = inputs2['responses']  # [batch, seq]
                len_responses = inputs2['len_responses']  # [batch, seq]
                sampled_latents = inputs2['sampled_latents']  # [batch, latent_size]
                id_responses_act = inputs2['post_emotion']
                len_decoder = id_responses.size(1) - 1

                embed_posts = word2vec.embedding(id_posts)  # [batch, seq, embed_size]
                embed_responses = word2vec.embedding(id_responses)  # [batch, seq, embed_size]
                # state: [layers, batch, dim]
                _, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
                _, state_responses = self.response_encoder(embed_responses.transpose(0, 1), len_responses)
                if isinstance(state_posts, tuple):
                    state_posts = state_posts[0]
                if isinstance(state_responses, tuple):
                    state_responses = state_responses[0]
                x = state_posts[-1, :, :]  # [batch, dim]
                y = state_responses[-1, :, :]  # [batch, dim]

                # p(z|x,y)
                z_mu, z_sigma2_log = self.recognize_net(x, y)  # [batch, latent]
                # 重参数化
                z = z_mu + (0.5 * z_sigma2_log).exp() * sampled_latents  # [batch, latent]

                z_ = z.detach().cpu().numpy()
                z_s.append(z_)
                label_s.append(id_responses_act.detach().cpu().numpy())

        z_s = np.concatenate(z_s, 0)
        label_s = np.concatenate(label_s, 0)
        return z_s, label_s

    def print_parameters(self):
        r""" 统计参数 """
        total_num = 0  # 参数总数
        for param in self.parameters():
            num = 1
            if param.requires_grad:
                size = param.size()
                for dim in size:
                    num *= dim
            total_num += num
        print(f"参数总数: {total_num}")

    def save_model(self, epoch, global_step, path):
        r""" 保存模型 """
        torch.save({'embedding': self.embedding.state_dict(),
                    'post_encoder': self.post_encoder.state_dict(),
                    'response_encoder': self.response_encoder.state_dict(),
                    'prior_net': self.prior_net.state_dict(),
                    'recognize_net': self.recognize_net.state_dict(),
                    'prepare_state': self.prepare_state.state_dict(),
                    'no_emotion_prepare_state': self.no_emotion_prepare_state.state_dict(),
                    'positive_prepare_state': self.positive_prepare_state.state_dict(),
                    'negative_prepare_state': self.negative_prepare_state.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'projector': self.projector.state_dict(),
                    'no_emotion_decoder': self.no_emotion_decoder.state_dict(),
                    'positive_decoder': self.positive_decoder.state_dict(),
                    'negative_decoder': self.negative_decoder.state_dict(),
                    'classification': self.classification.state_dict(),
                    'projector_no_emotion': self.projector_no_emotion.state_dict(),
                    'projector_positive': self.projector_positive.state_dict(),
                    'projector_negative': self.projector_negative.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step}, path)

    def load_model(self, path):
        r""" 载入模型 """
        checkpoint = torch.load(path)
        self.embedding.load_state_dict(checkpoint['embedding'])
        self.post_encoder.load_state_dict(checkpoint['post_encoder'])
        self.response_encoder.load_state_dict(checkpoint['response_encoder'])
        self.prior_net.load_state_dict(checkpoint['prior_net'])
        self.recognize_net.load_state_dict(checkpoint['recognize_net'])
        self.prepare_state.load_state_dict(checkpoint['prepare_state'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.projector.load_state_dict(checkpoint['projector'])
        self.projector_no_emotion.load_state_dict(checkpoint['projector_no_emotion'])
        self.projector_positive.load_state_dict(checkpoint['projector_positive'])
        self.projector_negative.load_state_dict(checkpoint['projector_negative'])
        self.no_emotion_decoder.load_state_dict(checkpoint['no_emotion_decoder'])
        self.positive_decoder.load_state_dict(checkpoint['positive_decoder'])
        self.negative_decoder.load_state_dict(checkpoint['negative_decoder'])
        self.classification.load_state_dict(checkpoint['classification'])
        self.no_emotion_prepare_state.load_state_dict(checkpoint['no_emotion_prepare_state'])
        self.positive_prepare_state.load_state_dict(checkpoint['positive_prepare_state'])
        self.negative_prepare_state.load_state_dict(checkpoint['negative_prepare_state'])
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        return epoch, global_step

    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.config.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)




    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

    def prepare_feed_data(self,data, inference=False):
        len_labels = torch.tensor([l - 1 for l in data['len_responses']]).long()  # [batch] 标签没有start_id，长度-1
        masks = (1 - F.one_hot(len_labels, len_labels.max() + 1).cumsum(1))[:, :-1]  # [batch, len_decoder]
        batch_size = masks.size(0)

        if not inference:  # 训练时的输入
            feed_data = {'posts': torch.tensor(data['posts']).long(),  # [batch, len_encoder]
                         'len_posts': torch.tensor(data['len_posts']).long(),  # [batch]
                         'responses': torch.tensor(data['responses']).long(),  # [batch, len_decoder]
                         'len_responses': torch.tensor(data['len_responses']).long(),  # [batch]
                         'sampled_latents': torch.randn((batch_size, self.config.latent_size)),  # [batch, latent_size]
                         'masks': masks.float(),
                         'post_emotion': torch.tensor(data['post_emotion']).long()}  # [batch, len_decoder]
        else:  # 测试时的输入
            feed_data = {'posts': torch.tensor(data['posts']).long(),
                         'len_posts': torch.tensor(data['len_posts']).long(),
                         'sampled_latents': torch.randn((batch_size, self.config.latent_size))}

        if True:  # 将数据转移到gpu上
            for key, value in feed_data.items():
                feed_data[key] = value.cuda()

        return feed_data

def cluster_acc(Y_pred, Y):
    # from scipy.optimize import linear_sum_assignment as linear_assignment
    from .util.linear_assignment_ import linear_assignment
    # 添加as语句不用修改代码中的函数名
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w

