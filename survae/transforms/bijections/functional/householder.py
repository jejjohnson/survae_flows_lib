import torch


def householder_matrix(V: torch.Tensor, loop: bool=False) -> torch.Tensor:
    n_reflections, num_dimensions = V.shape

    I = torch.eye(num_dimensions, dtype=V.dtype, device=V.device)

    Q = I

    if loop:
        for i in range(n_reflections):
            v = V[i].view(num_dimensions, 1)

            vvT = torch.matmul(v, v.t())
            vTv = torch.matmul(v.t(), v)
            Q = torch.matmul(Q, I - 2 * vvT / vTv)
    else:

        V_t = V.unsqueeze(2).transpose(1, 2)
        V = V.unsqueeze(2)
        I = torch.eye(num_dimensions, dtype=V.dtype, device=V.device)

        U = I - 2 * torch.bmm(V, V_t) / torch.bmm(V_t, V)
        Q = torch.linalg.multi_dot(tuple(U))


    return Q


def householder_matrix_fast(v: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """
    Fast product of a series of Householder matrices. This implementation is oriented to the one introducesd in:
    https://invertibleworkshop.github.io/accepted_papers/pdfs/10.pdf
    This makes use of method 2 in: https://ecommons.cornell.edu/bitstream/handle/1813/6521/85-681.pdf?sequence=1&isAllowed=y
    :param v: Batched series of Householder matrices. The last dim is the dim of one vector and the second last is the
    number of elements in one product. This is the min amount of dims that need to be present.
    All further ones are considered batch dimensions.
    :param stride: Controls the number of parallel operations by the WY representation (see paper)
    should not be larger than half the number of matrices in one product.
    :return: The batched product of Householder matrices defined by v
    """
    assert v.ndim > 1
    assert stride <= v.shape[-2]

    d, m = v.shape[-2], v.shape[-1]
    k = d // stride
    last = k * stride
    v = v / torch.norm(v, dim=-1, p=2, keepdim=True)
    v = v.unsqueeze(-1)
    u = 2 * v
    ID = torch.eye(m, device=u.device)
    for dim in range(v.ndim - 3):
        ID = ID.unsqueeze(0)

    # step 1 (compute intermediate groupings P_i)
    W = u[..., 0:last:stride, :, :]
    Y = v[..., 0:last:stride, :, :]

    for idx in range(1, stride):
        Pt = ID - torch.matmul(
            u[..., idx:last:stride, :, :],
            v[..., idx:last:stride, :, :].transpose(-1, -2),
        )
        W = torch.cat([W, u[..., idx:last:stride, :, :]], dim=-1)
        Y = torch.cat([torch.matmul(Pt, Y), v[..., idx:last:stride, :, :]], dim=-1)

    # step 2 (multiply the WY reps)
    P = ID - torch.matmul(W[..., k - 1, :, :], Y[..., k - 1, :, :].transpose(-1, -2))
    for idx in reversed(range(0, k - 1)):
        P = P - torch.matmul(
            W[..., idx, :, :], torch.matmul(Y[..., idx, :, :].transpose(-1, -2), P)
        )

    # deal with the residual, using a stride of 2 here maxes the amount of parallel ops
    if d > last:
        even_end = d if (d - last) % 2 == 0 else d - 1
        W_resi = u[..., last:even_end:2, :, :]
        Y_resi = v[..., last:even_end:2, :, :]
        for idx in range(last + 1, d if d == last + 1 else last + 2):
            Pt = ID - torch.matmul(
                u[..., idx:even_end:2, :, :],
                v[..., idx:even_end:2, :, :].transpose(-1, -2),
            )
            W_resi = torch.cat([W_resi, u[..., idx:even_end:2, :, :]], dim=-1)
            Y_resi = torch.cat(
                [torch.matmul(Pt, Y_resi), v[..., idx:even_end:2, :, :]], dim=-1
            )

        for idx in range(0, W_resi.shape[-3]):
            P = P - torch.matmul(
                P,
                torch.matmul(
                    W_resi[..., idx, :, :], Y_resi[..., idx, :, :].transpose(-1, -2)
                ),
            )

        if even_end != d:
            P = P - torch.matmul(
                P, torch.matmul(u[..., -1, :, :], v[..., -1, :, :].transpose(-1, -2))
            )

    return P
