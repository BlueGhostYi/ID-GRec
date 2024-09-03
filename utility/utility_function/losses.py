import torch
import math

def get_bpr_loss(user_embedding, positive_embedding, negative_embedding):

    pos_score = torch.sum(torch.mul(user_embedding, positive_embedding), dim=1)
    neg_score = torch.sum(torch.mul(user_embedding, negative_embedding), dim=1)

    # loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))

    loss = - torch.log(torch.sigmoid(pos_score - neg_score) + 10e-8)

    return torch.mean(loss)


def get_reg_loss(*embeddings):
    reg_loss = 0
    for embedding in embeddings:
        reg_loss += 1 / 2 * embedding.norm(2).pow(2) / float(embedding.shape[0])

    return reg_loss


def get_InfoNCE_loss(embedding_1, embedding_2, temperature):
    embedding_1 = torch.nn.functional.normalize(embedding_1)
    embedding_2 = torch.nn.functional.normalize(embedding_2)

    pos_score = (embedding_1 * embedding_2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)

    ttl_score = torch.matmul(embedding_1, embedding_2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

    cl_loss = - torch.log(pos_score / ttl_score + 10e-6)
    return torch.mean(cl_loss)


def get_InfoNCE_loss_all(embedding_1, embedding_2, embedding_2_all, temperature):
    embedding_1 = torch.nn.functional.normalize(embedding_1)
    embedding_2 = torch.nn.functional.normalize(embedding_2)
    embedding_2_all = torch.nn.functional.normalize(embedding_2_all)

    pos_score = (embedding_1 * embedding_2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)

    ttl_score = torch.matmul(embedding_1, embedding_2_all.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

    cl_loss = - torch.log(pos_score / ttl_score + 10e-8)
    return torch.mean(cl_loss)


def get_ELBO_loss(recon_x, x, mu, logvar, anneal):
    BCE = - torch.mean(torch.sum(torch.nn.functional.log_softmax(recon_x, 1) * x, -1))
    KLD = - 0.5 / recon_x.size(0) * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE, anneal * KLD


def get_align_loss(embedding_1, embedding_2):
    embedding_1 = torch.nn.functional.normalize(embedding_1, dim=-1)
    embedding_2 = torch.nn.functional.normalize(embedding_2, dim=-1)
    return torch.mean((embedding_1 - embedding_2).norm(p=2, dim=1).pow(2))


def get_uniform_loss(embedding):
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return torch.pdist(embedding, p=2).pow(2).mul(-2).exp().mean().log()
