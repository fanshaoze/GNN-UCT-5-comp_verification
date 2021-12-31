import _thread
import copy
import json
import numpy as np
import threading
import multiprocessing
import os
import time
import sys
import math
import random
from ucts import uct
from ucts import TopoPlanner
import datetime
from utils.util import mkdir, get_sim_configs, save_reward_hash, get_steps_traj, read_approve_path, \
    read_joint_component_prob
from utils.eliminate_isomorphism import unblc_comp_set_mapping, get_component_priorities
from SimulatorAnalysis import UCT_data_collection
from SimulatorAnalysis.simulate_with_topology import *
from GNN_gendata.GenerateTrainData import update_dataset
import gc


def merge_act_nodes(dest_act_node, act_node):
    dest_act_node.avg_return_ = dest_act_node.avg_return_ * dest_act_node.num_visits_ + \
                                act_node.avg_return_ * act_node.num_visits_
    dest_act_node.num_visits_ += act_node.num_visits_
    dest_act_node.avg_return_ = dest_act_node.avg_return_ / dest_act_node.num_visits_


def get_action_from_trees(uct_tree_list, uct_tree, tree_num=4):
    contain_action_flag = 0
    uct_tree.root_.act_vect_ = []
    for i in range(tree_num):
        for j in range(len(uct_tree_list[i].node_vect_)):
            contain_action_flag = 0
            for x in range(len(uct_tree.root_.act_vect_)):
                if uct_tree.root_.act_vect_[x].equal(uct_tree_list[i].act_vect_[j]):
                    contain_action_flag = 1
                    break
            if contain_action_flag == 1:
                if j < len(uct_tree_list[i].node_vect_) and uct_tree_list[i].node_vect_[j] is not None:
                    merge_act_nodes(uct_tree.root_.node_vect_[x], uct_tree_list[i].node_vect_[j])
            else:
                if j < len(uct_tree_list[i].node_vect_) and uct_tree_list[i].node_vect_[j] is not None:
                    uct_tree.root_.act_vect_.append(uct_tree_list[i].act_vect_[j].duplicate())
                    uct_tree.root_.node_vect_.append(uct_tree_list[i].node_vect_[j])
    act_node = uct_tree.get_action()
    return act_node


def get_action_from_planners(uct_planner_list, uct_tree, tree_num=4):
    contain_action_flag = 0
    uct_tree.root_.act_vect_ = []
    uct_tree.root_.node_vect_ = []
    for i in range(tree_num):
        for j in range(len(uct_planner_list[i].root_.node_vect_)):
            contain_action_flag = 0
            for x in range(len(uct_tree.root_.act_vect_)):
                if uct_tree.root_.act_vect_[x].equal(uct_planner_list[i].root_.act_vect_[j]):
                    contain_action_flag = 1
                    break
            if contain_action_flag == 1:
                if j < len(uct_planner_list[i].root_.node_vect_) and \
                        uct_planner_list[i].root_.node_vect_[j] is not None:
                    merge_act_nodes(uct_tree.root_.node_vect_[x], uct_planner_list[i].root_.node_vect_[j])
            else:
                if j < len(uct_planner_list[i].root_.node_vect_) and \
                        uct_planner_list[i].root_.node_vect_[j] is not None:
                    uct_tree.root_.act_vect_.append(uct_planner_list[i].root_.act_vect_[j].duplicate())
                    uct_tree.root_.node_vect_.append(uct_planner_list[i].root_.node_vect_[j])
    act_node = uct_tree.get_action()
    return act_node


def get_action_from_trees_vote(uct_planner_list, uct_tree, tree_num=4):
    action_nodes = []
    counts = {}
    for i in range(tree_num):
        action_nodes.append(uct_planner_list[i].get_action())
    for i in range(len(action_nodes)):
        tmp_count = 0
        if counts.get(action_nodes[i]) is None:
            for j in range(len(action_nodes)):
                if action_nodes[j].equal(action_nodes[i]):
                    tmp_count += 1
            counts[action_nodes[i]] = tmp_count
    for action, tmp_count in counts.items():
        if tmp_count == max(counts.values()):
            selected_action = action
    return selected_action


def read_DP_files(configs):
    target_min_vout = -500
    target_max_vout = 500
    if target_min_vout < configs['target_vout'] < 0:
        approved_path_freq = read_approve_path(0.0,
                                               './UCT_5_UCB_unblc_restruct_DP_v1/3comp_buck_boost_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None,
                                                             "./UCT_5_UCB_unblc_restruct_DP_v1/3comp_buck_boost_sim_node_joint_probs.json")

        print(approved_path_freq)
        print(component_condition_prob)

    elif 0 < configs['target_vout'] < 100:
        approved_path_freq = read_approve_path(0.0, './UCT_5_UCB_unblc_restruct_DP_v1/3comp_buck_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None,
                                                             "./UCT_5_UCB_unblc_restruct_DP_v1/3comp_buck_sim_node_joint_probs.json")

        print(approved_path_freq)
        print(component_condition_prob)


    elif 100 < configs['target_vout'] < target_max_vout:
        approved_path_freq = read_approve_path(0.0, './UCT_5_UCB_unblc_restruct_DP_v1/3comp_boost_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None,
                                                             "./UCT_5_UCB_unblc_restruct_DP_v1/3comp_boost_sim_node_joint_probs.json")
        print(approved_path_freq)
        print(component_condition_prob)
    else:
        return None
    return approved_path_freq, component_condition_prob


def write_info_to_file(fo, sim, effis, avg_cumulate_reward, avg_steps, total_query, total_hash_query, start_time,
                       tree_size, configs):
    fo.write("Final topology of game " + ":\n")
    fo.write("component_pool:" + str(sim.current.component_pool) + "\n")
    fo.write(str(sim.current.parameters) + "\n")
    fo.write("port_pool:" + str(sim.current.port_pool) + "\n")
    fo.write("graph:" + str(sim.current.graph) + "\n")
    fo.write("efficiency:" + str(effis) + "\n")
    fo.write("final reward:" + str(avg_cumulate_reward) + "\n")
    fo.write("step:" + str(avg_steps) + "\n")
    fo.write("query time:" + str(total_query) + " total tree size:" + str(tree_size) + "\n")
    fo.write("hash query time:" + str(total_hash_query) + "\n")
    end_time = datetime.datetime.now()
    fo.write("end at:" + str(end_time) + "\n")
    fo.write("start at:" + str(start_time) + "\n")
    fo.write("execute time:" + str((end_time - start_time).seconds) + " seconds\n")
    fo.write("result with parameter:" + str(sim.current.parameters) + "\n")
    fo.write("----------------------------------------------------------------------" + "\n")
    fo.write("configs:" + str(configs) + "\n")
    fo.write("final rewards:" + str(avg_cumulate_reward) + "\n")
    return fo


def print_and_write_child_info(fo, child_idx, _root, child, child_node, child_state):
    print("action ", child_idx, " :", _root.act_vect_[child_idx].type, "on",
          _root.act_vect_[child_idx].value)
    print("action child ", child_idx, " avg_return:", child.avg_return_)
    print("state child ", child_idx, " reward:", child_node.reward_)
    print("state ", child_idx, "ports:", child_state.port_pool)
    print("state child", child_idx, "graph:", child_state.graph)

    fo.write("action " + str(child_idx) + " :" + str(_root.act_vect_[child_idx].type) + "on" +
             str(_root.act_vect_[child_idx].value) + "\n")
    fo.write("action child " + str(child_idx) + " avg_return:" + str(child.avg_return_) + "\n")
    fo.write("action child " + str(child_idx) + " num_visits_:" + str(child.num_visits_) + "\n")
    fo.write(
        "state child " + str(child_idx) + "child node reward:" + str(child_node.reward_) + "\n")
    fo.write("state child " + str(child_idx) + "ports:" + str(child_state.port_pool) + "\n")
    fo.write("state child " + str(child_idx) + "graph:" + str(child_state.graph) + "\n")
    return fo


def get_multi_k_sim_results(sim, uct_simulators, configs, total_query, avg_query_time, avg_query_number,
                            anal_results, simu_results, save_tops, filted_save_tops, filter_sim):
    effis = []
    max_sim_reward_results = {}
    for k in configs['topk_list']:
        max_sim_reward_results[k] = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': -1,
                                     'max_sim_effi': -1, 'max_sim_vout': -500}

    sim.current, sim.reward, sim.current.parameters = sim.get_max_seen()
    if sim.current is None:
        return max_sim_reward_results
    max_result = sim.reward
    if configs['reward_method'] == 'analytics':
        max_topk = copy.deepcopy(uct_simulators[0].topk)
        for k in configs['topk_list']:
            sim.topk = copy.deepcopy(max_topk[-k:])
            # effis: [reward, effi, vout, para], get the max's reward
            effis = sim.get_reward_using_anal()

            if len(sim.topk) == 0:
                max_sim_reward_result = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': [],
                                         'max_sim_effi': -1, 'max_sim_vout': -500}
            else:
                max_sim_reward_result = get_simulator_tops_sim_info(sim=sim, filter_sim=filter_sim,
                                                                    thershold=configs['reward_threshold'])
                # [state, anal_reward, anal_para, key, sim_reward, sim_effi, sim_vout, sim_para]
                top_simus = []
                filted_top_simus = []
                for top in sim.topk:
                    top_simus.append(top[4])
                save_tops.append(top_simus)
                for top in sim.topk:
                    filted_top_simus.append(top[-1])
                filted_save_tops.append(filted_top_simus)

            max_sim_reward_results[k] = max_sim_reward_result
            if effis is not None:
                # anal_result [effi, vout, max result, para, total query]
                anal_result = [effis[1], effis[2], max_result, str(sim.current.parameters), total_query,
                               avg_query_time, avg_query_number]
                anal_results[k].append(anal_result)
            simu_results[k].append(
                [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                 max_sim_reward_result['max_sim_reward'],
                 max_sim_reward_result['max_sim_para'],
                 total_query, avg_query_time, avg_query_number])

    elif configs['reward_method'] == 'simulator':
        effis = sim.get_reward_using_sim()
        max_sim_reward_result = {}
        if len(sim.topk) == 0:
            max_sim_reward_result = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': -1,
                                     'max_sim_effi': -1, 'max_sim_vout': -500}
        else:
            max_sim_reward_result = get_simulator_tops_sim_info(sim=sim)
        for k in configs['topk_list']:
            if effis is not None:
                # anal_result [effi, vout, max result, para, total query]
                anal_results[k].append([effis[1], effis[2], max_result, str(sim.current.parameters),
                                        total_query, avg_query_time, avg_query_number])
            simu_results[k].append(
                [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                 max_sim_reward_result['max_sim_reward'],
                 max_sim_reward_result['max_sim_para'],
                 total_query, avg_query_time, avg_query_number])
    else:
        effis = None
    print("effis of topo:", effis)

    return anal_results, simu_results, effis, save_tops, filted_save_tops


def trajs_all_in_first_step(total_step, num_runs):
    steps_traj = []
    for i in range(total_step):
        if i == 0:
            steps_traj.append(total_step * num_runs - (total_step - 1))
        else:
            steps_traj.append(1)
    return steps_traj


def copy_simulators_info(sim, uct_simulators):
    sim.graph_2_reward = uct_simulators[0].graph_2_reward
    sim.current_max = uct_simulators[0].current_max
    sim.no_isom_seen_state_list = uct_simulators[0].no_isom_seen_state_list
    sim.key_expression = uct_simulators[0].key_expression
    sim.key_sim_effi_ = uct_simulators[0].key_sim_effi_
    sim.new_query_time += uct_simulators[0].new_query_time
    sim.new_query_counter += uct_simulators[0].new_query_counter
    sim.topk = uct_simulators[0].topk
    if hasattr(sim, 'surrogate_hash_table') and hasattr(uct_simulators[0], 'surrogate_hash_table'):
        sim.surrogate_hash_table = uct_simulators[0].surrogate_hash_table
    return sim


def get_total_querys(sim, uct_simulators):
    total_query = sim.query_counter
    total_hash_query = sim.hash_counter
    for simulator in uct_simulators:
        total_query += simulator.query_counter
        total_hash_query += simulator.hash_counter
    return total_query, total_hash_query


def pre_fix_topo(sim):
    # For fixed commponent type
    init_nodes = []
    init_edges = []
    # init_nodes = [0,1,2,1,2]
    # init_edges = [[0,9],[1,11],[2,3],[3,7],[4,12],[5,12],[6,8],[10,12]]
    print(init_nodes)
    # init_nodes = [1, 0, 1, 3, 0]
    # init_nodes = [0, 0, 3, 3, 1]
    # init_nodes = [0, 0, 1, 2, 3]
    for node in init_nodes:
        action = TopoPlanner.TopoGenAction('node', node)
        sim.act(action)

    # edges = [[0, 7], [1,10], [2,6], [3,9], [4, 8], [5,9], [-1,-1], [-1,-1]]
    # edges = [[0, 6], [1,3], [2,10], [3,7], [4, 11], [5,8], [6,12], [-1,-1], [8,11],[9,11]
    #     , [-1,-1],[-1,-1],[-1,-1]]

    # edges = [[0, 3], [1, 8], [2, 5], [4, 7], [6, 7]]
    # edges = [[0, 8], [1, 3], [2, 4], [4, 6], [5, 7]]
    for edge in init_edges:
        action = TopoPlanner.TopoGenAction('edge', edge)
        sim.act(action)

    # for action_tmp in sim.act_vect:
    #     print(action_tmp.value)
    return init_nodes, init_edges, sim


def compute_statistic(preds, ground_truth, good_topo_threshold=0.6):
    """
    Generate TPR, FPR points under different thresholds for the ROC curve.
    Reference: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
    :param preds: a list of surrogate model predictions
    :param ground_truth: a list of ground-truth values (e.g. by the simulator)
    :return: {threshold: {'TPR': true positive rate, 'FPR': false positive rate}}
    """
    preds, ground_truth = np.array(preds), np.array(ground_truth)

    thresholds = np.arange(0.001, 1., step=0.001)  # (0.1, 0.2, ..., 0.9)
    result = {}

    for thres in thresholds:
        good = np.where(ground_truth >= good_topo_threshold)[0]
        G_count = len(good)
        gt_pos = np.where(ground_truth >= thres)[0]
        P_count = len(gt_pos)
        gt_neg = np.where(ground_truth < thres)[0]
        N_count = len(gt_neg)
        predict_pos = np.where(preds >= thres)[0]
        PP_count = len(predict_pos)
        predict_neg = np.where(preds < thres)[0]
        PN_count = len(predict_neg)

        if len(gt_pos) == 0:
            TPR, TP_max, TP_count, TP_good_count, TP_good_rate, TP_good_whole_rate = 0, 0, 0, 0, 0, 0
        else:
            TP = np.intersect1d(gt_pos, predict_pos)
            TP_count = len(TP)
            TPR = TP_count / P_count
            if TP_count == 0:
                TP_max, TP_good_count, TP_good_rate, TP_good_whole_rate = 0, 0, 0, 0
            else:
                TP_max = max([ground_truth[idx] for idx in TP])
                TP_good = [ground_truth[idx] for idx in TP if ground_truth[idx] >= good_topo_threshold]
                TP_good_count = len(TP_good)
                TP_good_rate = TP_good_count / TP_count
                TP_good_whole_rate = TP_good_count / G_count

        if len(gt_pos) == 0:
            FNR, FN_max, FN_count, FN_good_count, FN_good_rate, FN_good_whole_rate = 0, 0, 0, 0, 0, 0
        else:
            FN = np.intersect1d(gt_pos, predict_neg)
            FN_count = len(FN)
            FNR = len(FN) / len(gt_pos)
            if FN_count == 0:
                FN_max, FN_good_count, FN_good_rate, FN_good_whole_rate = 0, 0, 0, 0
            else:
                FN_max = max([ground_truth[idx] for idx in FN])
                FN_good = [ground_truth[idx] for idx in FN if ground_truth[idx] >= good_topo_threshold]
                FN_good_count = len(FN_good)
                FN_good_rate = FN_good_count / FN_count
                FN_good_whole_rate = FN_good_count / G_count

        if len(gt_neg) == 0:
            FPR = 0
        else:
            FPR = len(np.intersect1d(gt_neg, predict_pos)) / len(gt_neg)

        result[thres] = dict(P_count=P_count, N_count=N_count, PP_count=PP_count, PN_count=PN_count,
                             TP_count=TP_count, TPR=TPR, TP_max=TP_max, TP_good_count=TP_good_count,
                             TP_good_rate=TP_good_rate, TP_good_whole_rate=TP_good_whole_rate,
                             FN_count=FN_count, FNR=FNR, FN_max=FN_max, FN_good_count=FN_good_count,
                             FN_good_rate=FN_good_rate, FN_good_whole_rate=FN_good_whole_rate,
                             FPR=FPR)

    return result


def write_results_to_csv_file(file_name, results):
    """
    @param file_name:
    @param results: {k:{'max':max, 'mean':mean, 'std':std, 'precision', precision, 'recall', recall}}
    @return:
    """

    output_for_ks = []
    index = 0
    for k, values in results.items():
        csv_list = [k]
        csv_list.extend(list(values.values()))
        output_for_ks.append(csv_list)
        header = ['k']
        header.extend([key for key in values])
    with open(file_name, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerows(output_for_ks)
    f.close()


def save_raw_data_to_files(file_name, raw_data):
    with open(file_name + '.json', 'w') as f:
        json.dump(raw_data, f)
    f.close()

    raw_data_csv = []
    for i in range(len(raw_data['pred_rewards'])):
        raw_data_csv.append([raw_data['pred_rewards'][i], raw_data['ground_truth'][i]])
    header = ['pred_rewards', 'ground_truth']
    with open(file_name + '.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerows(raw_data_csv)
    f.close()


def class_data_and_save_to_files(class_data_file_name, raw_data, class_number=10):
    class_raw_data = {}
    class_csv_raw_data = []
    header = []
    for i in range(class_number):
        class_raw_data[i * 1 / class_number] = [data for data in raw_data if
                                                i * 1 / class_number <= data < (i + 1) * 1 / class_number]
        class_csv_raw_data.append([data for data in raw_data if
                                   i * 1 / class_number <= data < (i + 1) * 1 / class_number])
        header.append(str(format(i * 1 / class_number, '.3f')) +
                      '-' + str(format((i + 1) * 1 / class_number, '.3f')))
    with open(class_data_file_name + '.json', 'w') as f:
        json.dump(class_raw_data, f)
    f.close()
    with open(class_data_file_name + '.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerows(class_csv_raw_data)
    f.close()
    return class_raw_data


def filter_test(trajectory, test_number, configs, result_folder, Sim=None, uct_tree_list=None, keep_uct_tree=False):
    global component_condition_prob
    if Sim is None:
        Sim = TopoPlanner.TopoGenSimulator
        inside_sim = True
    else:
        inside_sim = False
    # path = './UCT_5_UCB_unblc_restruct_DP_v1/SimulatorAnalysis/database/analytic-expression.json'
    # is_exits = os.path.exists(path)
    # if not is_exits:
    #     UCT_data_collection.key_expression_dict()
    # print("finish reading key-expression")
    out_file_folder = 'Results/' + result_folder + '/'
    mkdir(out_file_folder)
    out_file_name = out_file_folder + str(trajectory) + '-result.txt'
    out_round_folder = 'Results/' + result_folder + '/' + str(trajectory)
    mkdir(out_round_folder)
    figure_folder = "figures/" + result_folder + "/"
    mkdir(figure_folder)

    # out_file_name = "Results/mutitest_" + str(configs['target_vout']) + "-" + date_str + "-" + str(os.getpid()) + ".txt"
    # figure_folder = "figures/" + result_folder + "/"
    # mkdir(figure_folder)

    sim_configs = get_sim_configs(configs)
    start_time = datetime.datetime.now()

    simu_results = {}
    anal_results = {}
    save_tops = []
    filted_save_tops = []
    for k in configs['topk_list']:
        simu_results[k] = [['efficiency', 'vout', 'reward', 'DC_para', 'query', 'avg_time', 'new_query']]
        anal_results[k] = [['efficiency', 'vout', 'reward', 'DC_para', 'query', 'avg_time', 'new_query']]

    fo = open(out_file_name, "w")
    fo.write("max_depth,num_runs,avg_step\n")
    avg_step_list = []

    for test_idx in range(test_number):
        fo.write("----------------------------------------------------------------------" + "\n")
        num_runs = trajectory
        avg_steps, avg_cumulate_reward, steps, cumulate_plan_time, r, tree_size, preset_component_num = \
            0, 0, 0, 0, 0, 0, 0
        cumulate_reward_list, uct_simulators, uct_tree_list = [], [], []

        approved_path_freq, component_condition_prob = read_DP_files(configs)
        key_expression = UCT_data_collection.read_no_sweep_analytics_result()
        key_sim_effi = UCT_data_collection.read_no_sweep_sim_result()

        component_priorities = get_component_priorities()
        # TODO must be careful, if we delete the random adding of sa,sb,
        #  we also need to change the preset comp number
        _unblc_comp_set_mapping, _ = unblc_comp_set_mapping(['Sa', 'Sb', 'L', 'C'],
                                                            configs['num_component'] - 3 - preset_component_num)
        # for k, v in _unblc_comp_set_mapping.items():
        #     print(k, '\t', v)

        # init outer simulator and tree

        sim = Sim(sim_configs, approved_path_freq,
                  component_condition_prob,
                  key_expression, _unblc_comp_set_mapping, component_priorities,
                  key_sim_effi,
                  None, configs['num_component'])

        isom_topo_dict = {}
        dataset_isom_topo_dict = {}
        statisitc_results = {}
        class_number = 100
        k = 10000

        model_list = ['reg_reward-8-4', 'reg_reward-5-0']
        # for model_name in model_list:
        #     file_name = model_name + '_' + str(k) + '_statistic.json'
        #
        #     result = json.load(open(file_name))
        #     write_results_to_csv_file(file_name.replace('.json', '.csv'), result)
        # return

        while k > 0:
            sim_train = Sim(sim_configs, approved_path_freq,
                            component_condition_prob,
                            key_expression, _unblc_comp_set_mapping, component_priorities,
                            key_sim_effi,
                            None, configs['num_component'])
            isom_topo, key = sim_train.generate_random_topology_without_reward()
            if not isom_topo.graph_is_valid():
                continue
            anal_reward = sim_train.get_reward()

            if key not in isom_topo_dict:
                isom_topo_dict[key] = {'anal_reward': anal_reward, 'topo': isom_topo}
                k -= 1
            print('--------------------------------------', k)
            del sim_train
        anal_rewards = [isom_topo_dict[key]['anal_reward'] for key in isom_topo_dict]
        class_data_file_name = 'analytic' + '-' + str(k) + '-' + '-class-' + str(class_number)
        class_data_and_save_to_files(class_data_file_name=class_data_file_name,
                                     raw_data=anal_rewards, class_number=class_number)
        for model_name in model_list:
            gnn_rewards = []
            configs['reward_model'] = model_name
            sim_configs = get_sim_configs(configs)
            from topo_envs.GNNRewardSim import GNNRewardSim
            def sim_init(*a):
                return GNNRewardSim(configs['eff_model'], configs['vout_model'], configs['reward_model'],
                                    configs['debug'], *a)

            filter_sim = sim_init(sim_configs, approved_path_freq,
                                  component_condition_prob,
                                  key_expression, _unblc_comp_set_mapping, component_priorities,
                                  key_sim_effi,
                                  None, configs['num_component'], None)

            for key, topo_info in isom_topo_dict.items():
                # self, _next_candidate_components=None, _current_candidate_components=None, state=None
                filter_sim.set_state(None, None, topo_info['topo'])
                gnn_reward = filter_sim.get_reward()
                gnn_rewards.append(gnn_reward)

            pred_rewards = [gnn_raw_reward.item() for gnn_raw_reward in gnn_rewards]
            raw_data = {'pred_rewards': pred_rewards, 'ground_truth': anal_rewards}
            raw_data_file_name = model_name + '-' + str(k) + '-' + 'anal_as_gt'

            save_raw_data_to_files(file_name=raw_data_file_name, raw_data=raw_data)
            class_data_file_name = raw_data_file_name + '-class-' + str(class_number)
            class_data_and_save_to_files(class_data_file_name=class_data_file_name,
                                         raw_data=pred_rewards, class_number=class_number)

            result = compute_statistic(preds=gnn_rewards, ground_truth=anal_rewards, good_topo_threshold=0.6)
            statisitc_results[model_name] = result
            for threshold, statisitc in result.items():
                print(threshold, ' ', statisitc)

            with open(configs['reward_model'] + '_' + str(len(isom_topo_dict)) + '_statistic.json', 'w') as f:
                json.dump(result, f)
            f.close()

            del filter_sim
    end_time = datetime.datetime.now()
    print((end_time - start_time).seconds)

    return


'''
for key, topo_info in isom_topo_dict.items():
    if topo_info['gnn_reward'] < sim.configs_['reward_threshold']:
        # if True:
        print('filted: gnn:', topo_info['gnn_reward'], ' analytics: ', topo_info['anal_reward'])
        if topo_info['anal_reward'] > sim.configs_['reward_threshold']:
            # if True:  # just for test
            isom_topo = topo_info['topo']
            # topk_max_reward, eff, vout, topk_max_para = \
            #     get_single_topo_sim_result(current=isom_topo, sweep=sim.configs_['sweep'],
            #                                candidate_params=sim.candidate_params,
            #                                key_sim_effi_=sim.key_sim_effi_,
            #                                skip_sim=sim.configs_['skip_sim'],
            #                                key_expression_mapping=sim.key_expression,
            #                                target_vout=sim.configs_['target_vout'],
            #                                min_vout=sim.configs_['min_vout'])
            topk_max_reward, eff, vout, topk_max_para = 0, 0.9, 1, 0
            isom_topo.parameters = [0.5, 10, 100]
            dataset_isom_topo_dict[key + '$' + '[0.5, 10, 100]'] = [isom_topo, eff, vout,
                                                                    topo_info['anal_reward'], 50]

# date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# os.system('filter_dataset.json')
update_dataset(dataset_isom_topo_dict, training_date_file='filter_dataset.json')
'''
