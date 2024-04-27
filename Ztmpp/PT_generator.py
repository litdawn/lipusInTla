# generate_next中的选择动作，供参考
# self.depth = 0
#         PT = InitPT()  # Bool('non_nc')
#         # 1
#         self.stateVec = self.T(PT)
#         # the lists will be used when punish or prised.
#         predicted_reward_list = []
#         action_selected_list = []
#         outputed_list = []
#         action_or_value = []
#         left_handles = []
#         # CE = {'p': [],'n': [],'i': []}
#         # 2
#         emb_CE = self.E(CE)
#         # pre_exp, trans_exp, post_exp这三个是self.smt
#         # 3
#         self.emb_smt = self.T.forward_three(self.smt)
#         # 函数返回的是输入表达式 PT 中最左侧的句柄, non代表非终结符
#         left_handle = getLeftHandle(PT)
#         while left_handle is not None:
#             left_handles.append(left_handle)
#             # 在templatecenter的RULE里面随机选规则
#             act_or_val, available_acts = AvailableActionSelection(left_handle)
#             # 总体特征
#             overall_feature = self.G(self.emb_smt, emb_CE, self.stateVec)
#             predicted_reward = self.P(self.stateVec, overall_feature)
#             predicted_reward_list.append(predicted_reward)
#             action_vector = self.pi(self.stateVec, overall_feature)
#             if act_or_val == config.SELECT_AN_ACTION:
#                 action_dirtibution, action_raw = self.distributionlize(action_vector, available_acts)
#                 action_selected = sampling(action_dirtibution, available_acts)
#
#                 if self.depth >= config.MAX_DEPTH:
#                     action_selected = simplestAction(left_handle)
#                 action_selected_list.append(action_selected)
#                 outputed_list.append(action_raw)
#
#                 PT = update_PT_rule_selction(PT, left_handle, action_selected)
#
#             else:
#                 assert False  # should not be here now
#                 # value = self.intValuelzie(action_vector, left_handle)
#                 # value_of_int = int(value)
#                 # action_selected_list.append(value_of_int)
#                 # outputed_list.append(value)
#                 #
#                 # PT = update_PT_value(PT, left_handle, value_of_int)
#
#             action_or_value.append(act_or_val)
#             left_handle = getLeftHandle(PT)
#             self.stateVec = self.T(PT)
#             self.depth += 1
#
#         self.last_predicted_reward_list = predicted_reward_list
#         self.last_action_selected_list = action_selected_list
#         self.last_outputed_list = outputed_list
#         self.last_action_or_value = action_or_value
#         self.last_left_handles = left_handles
#         return PT

