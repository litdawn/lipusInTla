# def opt_quant_inv_fn(invarg):
#     # Remove unnecessary quantified variables if they do not appear in the
#     # unquantified expression. For example, we would simplify an expression like
#     #
#     #   \A i,j \in Server : i > 3
#     #
#     # to:
#     #
#     #   \A i \in Server : i > 3
#     #
#     # i.e. get rid of the extra quantified variable out front. This can
#     # make it faster for TLC to check candidate invariants.
#     quantifiers_to_keep = []
#     for q in var_quantifiers:
#         quant_var_name = q.split(" ")[1]
#         if quant_var_name in invarg:
#             quantifiers_to_keep.append(q)
#     return " : ".join(quantifiers_to_keep + [invarg])