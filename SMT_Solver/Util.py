import re


def chunks(seq, n_chunks):
    """ Splits a given iterable into n evenly (as possible) sized chunks."""
    N = len(seq)
    chunk_size = max(N // n_chunks, 1)
    # print("chunk size:", chunk_size)
    return (seq[i:i + chunk_size] for i in range(0, N, chunk_size))


def grep_lines(pattern, lines):
    return [ln for ln in lines if re.search(pattern, ln)]


# def get_invexp(inv):
#     invi = int(inv.replace("Inv", ""))
#     return quant_inv_fn(orig_invs_sorted[invi])
#
#
# def get_invexp_cost(inv):
#     exp = get_invexp(inv)
#     return exp

def print_cti_set(set_cti):
    for cti in set_cti:
        print(cti.getCTIStateString())

