import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85,0.64]]
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

att_weoghts_2_tmp = attn_scores_2 / attn_scores_2.sum()
print(att_weoghts_2_tmp)
print(att_weoghts_2_tmp.sum())

att_weoghts_2 = torch.softmax(attn_scores_2, dim=0)
print(att_weoghts_2)
print(att_weoghts_2.sum())

