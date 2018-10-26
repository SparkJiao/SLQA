import torch


def vector_weight_sum(weight_vector: torch.Tensor, matrix: torch.Tensor):
    # Shape of weight_vector: (batch_size, len)
    # Shape of matrix: (batch_size, n, len)
    # Shape of output vector:(batch_size, n)
    assert weight_vector.size(0) == matrix.size(0)
    assert weight_vector.size(1) == matrix.size(2)
    batch_size = weight_vector.size(0)
    out_features = matrix.size(1)
    len = weight_vector.size(1)
    output = torch.zeros(batch_size, out_features)
    matrix = matrix.permute(0, 2, 1)
    for i in range(batch_size):
        for j in range(len):
            output[i] = output[i] + torch.mul(weight_vector[i][j], matrix[i][j])
    return output


def vector_weight_sum_matrix(weight_vector_matrix: torch.Tensor, matrix: torch.Tensor):
    # Shape of weight_vector_matrix: (batch_size, m, len)
    # Shape of matrix: (batch_size, n, len)
    # Shape of output: (batch_size, n, m)
    assert weight_vector_matrix.size(0) == matrix.size(0)
    assert weight_vector_matrix.size(2) == matrix.size(2)
    batch_size = weight_vector_matrix.size(0)
    len = matrix.size(2)
    n = matrix.size(1)
    m = weight_vector_matrix.size(2)
    output = torch.zeros(batch_size, m, n)
    matrix.permute(0, 2, 1)
    for i in range(batch_size):
        for j in range(m):
            for k in range(len):
                output[i][j] = output[i][j] + torch.mul(weight_vector_matrix[i][j][k], matrix[i][k])
    output = output.permute(0, 2, 1)
    return output

def attention_weight_sum_batch(attention_vector, matrix):
    # shape of attention_vector: (batch_size, vector_num, vector_dim)
    # shape of matrix: (batch_size, text_length(token_nums), encoding_dim)
    # vector_dim = text_length
    # shape of output_matrixL (batch_size, vector_num, encoding_dim)
    return torch.bmm(attention_vector, matrix)
