import argparse
import sys
import os
import torch
import numpy as np
import random
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio import PDB, SeqIO
import scipy.stats as stats
sys.path.insert(0, '/cs/labs/dina/seanco/xl_parser')
import pdb_files_manager
import cross_link
import train
import graph_dataset
import data_proccess


device = train.device


def update_xlink_file_to_dimeric(in_file, out_file, chain_dimer_dict):
    in_f = open(in_file, 'r')
    out_f = open(out_file, 'w')
    for line in in_f:
        words = line.split()
        words[1] = words[1] + chain_dimer_dict[words[1]]
        words[3] = words[3] + chain_dimer_dict[words[3]]
        new_line = " ".join(words) + '\n'
        out_f.write(new_line)
    out_f.close()
    in_f.close()


def preprocess_xlink_file(in_file, out_file, dimers):
    chain_dimer_dict = dict()
    pairs = dimers.split()
    for pair in pairs:
        chain_dimer_dict[pair[0]] = pair[1]
        chain_dimer_dict[pair[1]] = pair[0]
    update_xlink_file_to_dimeric(in_file, out_file, chain_dimer_dict)


def print_pred_to_file(obj, min_dist, max_dist, y_pred_prob, out_file, dimer_dict=None):
    y_pred_prob = [str(item) for item in y_pred_prob]
    probs_str = " ".join(y_pred_prob)
    if dimer_dict is None:
        chain_a = obj.uniport_a
        chain_b = obj.uniport_b
    else:
        chain_a = dimer_dict[obj.uniport_a]
        chain_b = dimer_dict[obj.uniport_b]
    out_file.write(f"{obj.res_num_a} {chain_a} {obj.res_num_b} {chain_b} {min_dist} {max_dist},"
                   f" {probs_str}\n")


def print_prediction(obj, label, cfg, y_pred_prob, validate=True, out_file=None, dimer_dict=None):
    min_dist, max_dist = graph_dataset.XlPairGraphDataset.get_dist_from_label(label, cfg['model_th'])
    print(f"Cross link: {obj.res_num_a} {obj.uniport_a} {obj.res_num_b} "
          f"{obj.uniport_b}\nPredicted range: {min_dist} - {max_dist} \nProbabilities: {y_pred_prob}")
    if validate:
        print(f"True distance in pdb: {obj.distance}")
    if out_file is not None:
        print_pred_to_file(obj, min_dist, max_dist, y_pred_prob, out_file, dimer_dict)


def create_fake_xl_objects(pdb_name, amount=50, linker='DSS', max_distance=45):
    uniport = pdb_name.split('/')[-1].split('.')[0]
    pdb_parser = PDBParser(PERMISSIVE=1)
    ppb = PDB.CaPPBuilder()
    structure = pdb_parser.get_structure(pdb_name.split('/')[-1], pdb_name)
    polypeptide_list = ppb.build_peptides(structure, 1)
    lys_residues = []
    xl_objets = []
    for pep in polypeptide_list:
        for res in pep:
            if res.resname == 'LYS':
                lys_residues.append(res)
    for i in range(len(lys_residues)):
        for j in range(i + 1, len(lys_residues)):
            obj = cross_link.CrossLink("", "", lys_residues[i].id[1], uniport, "", "",
                                       lys_residues[j].id[1], uniport, linker, uniport)
            xl_objets.append(obj)
    random.shuffle(xl_objets)
    for obj in xl_objets:
        obj.process_single_xl(polypeptide_list)
    xl_objets = [obj for obj in xl_objets if obj.distance <= max_distance]
    xl_objets = xl_objets[:amount]
    return xl_objets


def fake_predict(cfg, data_name, xl_objects=None, best_prob=0.8, rand_prob=0.1, validate=True,
                 out_file=None, dimer_dict=None):
    if xl_objects is None:
        xl_objects = create_fake_xl_objects(data_name)
    distances = [obj.distance for obj in xl_objects]
    distances = np.asarray(distances)
    labels, th = data_proccess.FeatDataset.get_labels_from_dist(np.copy(distances), cfg['num_classes'],
                                                                cfg['model_th'])
    labels = labels.astype(int)
    pred_probs = np.zeros((len(xl_objects), cfg['num_classes']))
    for i, obj in enumerate(xl_objects):
        true_label = labels[i]
        if true_label > 0:
            sec_label = true_label - 1
        else:
            sec_label = true_label + 1
        if true_label == cfg['num_classes'] - 1:
            third_label = true_label - 2
        else:
            third_label = true_label + 1
        p1 = random.uniform(0.5, 1)
        p2 = random.uniform(0, 1 - p1)
        p3 = 1 - (p1 + p2)
        p = random.uniform(0, 1)
        if p < best_prob:
            pred_probs[i][true_label] = p1
            pred_probs[i][sec_label] = p2
            pred_probs[i][third_label] = p3
        elif p > (1 - rand_prob):
            l1, l2, l3 = random.sample(range(cfg['num_classes']), 3)
            pred_probs[i][l1] = p1
            pred_probs[i][l2] = p2
            pred_probs[i][l3] = p3
        else:
            pred_probs[i][true_label] = p2
            pred_probs[i][sec_label] = p1
            pred_probs[i][third_label] = p3
        print_prediction(obj, labels[i], cfg, pred_probs[i], validate, out_file, dimer_dict)
    train.plot_cm(labels, pred_probs)
    return xl_objects, labels, pred_probs


def predict(cfg, xl_objects, validate=True, out_file=None, dimer_dict=None):
    model = train.get_model(cfg)
    model.load_state_dict(torch.load(f"models/{cfg['name']}"))
    model.to(device)
    model.eval()
    _, dataset = train.get_dataset(cfg)
    data_loader, _ = train.load_data(cfg, dataset, None, shuffle=False)
    probas = np.zeros((len(xl_objects), cfg['num_classes']))
    labels = np.zeros(len(xl_objects))
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs = data.to(device)
            outputs = model(inputs)
            y_pred_prob = torch.softmax(outputs[0], dim=1)
            _, y_pred_tags = torch.max(y_pred_prob, dim=1)
            labels[i] = (y_pred_tags.item())
            probas[i] = y_pred_prob.cpu().numpy()
            print_prediction(xl_objects[i], y_pred_tags.item(), cfg, y_pred_prob.cpu().tolist()[0],
                             validate, out_file, dimer_dict)
        return labels, probas


def initialize_obj_angles(obj):
    obj.omega = np.array([1, 1])
    obj.phi = np.array([1, 1, 1, 1])
    obj.theta = np.array([1, 1, 1, 1])


def create_xl_obj_in_dimer_xl(chain_a, chain_b, res_a, res_b, linker, objects, peptides, dimer_dict):
    # best_obj = None
    for c_a in chain_a:
        for c_b in chain_b:
            dimer_dict[c_a] = chain_a
            dimer_dict[c_b] = chain_b
            obj = cross_link.CrossLink("", "", res_a, c_a, "", "",
                                       res_b, c_b, linker)
            # obj.process_single_xl(peptides, True)
            obj.process_single_xl(None, True, peptides)
            initialize_obj_angles(obj)
            # if best_obj is None or 0 < obj.distance < best_obj.distance:
            #     best_obj = obj
            objects.append(obj)


def create_multichain_xl_obj(chain_a, chain_b, res_a, res_b, linker, objects, peptides, dimer_dict, validate):
    if validate and (len(chain_a) > 1 or len(chain_b) > 1):
        create_xl_obj_in_dimer_xl(chain_a, chain_b, res_a, res_b, linker, objects, peptides, dimer_dict)
    else:
        obj = cross_link.CrossLink("", "", res_a, chain_a[0], "", "", res_b, chain_b[0], linker)
        if validate:
            obj.process_single_xl(None, True, peptides)
        initialize_obj_angles(obj)
        objects.append(obj)
        dimer_dict[chain_a[0]] = chain_a
        dimer_dict[chain_b[0]] = chain_b


def create_xl_objects_multiple_pdb(xl_file, linker, solution_pdb=None, validate=False):
    objects = []
    dimer_dict = dict()
    peptides = None
    if validate and solution_pdb is not None:
        suff = solution_pdb.split('.')[-1]
        if suff == 'cif':
            pdb_parser = MMCIFParser()
        else:
            pdb_parser = PDBParser(PERMISSIVE=1)
        structure = pdb_parser.get_structure(solution_pdb.split('/')[-1], solution_pdb)
        peptides = list(structure.get_chains())
    with open(xl_file, 'r') as f:
        for line in f:
            res_a, chain_a, res_b, chain_b, _, _ = line.split(' ')
            create_multichain_xl_obj(chain_a, chain_b, res_a, res_b, linker, objects, peptides, dimer_dict, validate)
    return objects, dimer_dict


def create_xl_objects_single_pdb(xl_file, pdb_name, linker, validate=False, multichain=False):
    objects = []
    dimer_dict = dict()
    polypeptide_list, chains = None, None
    uniport = pdb_name.split('/')[-1].split('.')[0]
    if validate:
        pdb_parser = PDBParser(PERMISSIVE=1)
        ppb = PDB.CaPPBuilder()
        structure = pdb_parser.get_structure(pdb_name.split('/')[-1], pdb_name)
        polypeptide_list = ppb.build_peptides(structure, 1)
        chains = list(structure.get_chains())
    with open(xl_file, 'r') as f:
        for line in f:
            res_a, chain_a, res_b, chain_b, _, _ = line.split(' ')
            if not multichain:
                obj = cross_link.CrossLink("", "", res_a, uniport, "", "",
                                           res_b, uniport, linker, uniport)
                peptides = polypeptide_list
                if validate:
                    obj.process_single_xl(peptides, multichain)
                initialize_obj_angles(obj)
                objects.append(obj)
            else:
                peptides = chains
                create_multichain_xl_obj(chain_a, chain_b, res_a, res_b, linker, objects, peptides,
                                         dimer_dict, validate)
    return objects, dimer_dict


def update_feat_dict_by_dimers(feat_dict, dimer_dict):
    new_dict = dict()
    for key, val in feat_dict.items():
        if key in dimer_dict:
            for new_key in dimer_dict[key]:
                if key != new_key:
                    new_dict[new_key] = val
    feat_dict.update(new_dict)


def pre_process_data(xl_file, pdb_file, linker, solution_pdb, multichain):
    data_name = xl_file.split('/')[-1].split('.')[0]
    if len(pdb_file) == 1:
        xl_objects, dimer_dict = create_xl_objects_single_pdb(xl_file, pdb_file, linker,
                                                              multichain)
    else:
        xl_objects, dimer_dict = create_xl_objects_multiple_pdb(xl_file, linker,
                                                                solution_pdb, True)
    xl_feat_path = '/'.join(pdb_file[0].split('/')[:-1]) + '/xl_features/'
    if not os.path.isdir(xl_feat_path):
        os.makedirs(xl_feat_path)
    pdb_files_manager.single_thread_extract_xl_features([xl_file], [pdb_file], output_path=xl_feat_path)
    xl_files_path = [xl_feat_path + pdb.split('/')[-1].split('.')[0] + '.txt' for pdb in pdb_file]
    feat_dict = dict()
    pdb_files_manager.predict_read_features(pdb_file, feat_dict, xl_files_path)
    update_feat_dict_by_dimers(feat_dict, dimer_dict)
    graph_dataset.generate_graph_data(xl_objects, feat_dict, None, None, data_name, pdb_file)
    return data_name, xl_objects, dimer_dict


def parse_args():
    parser = TrainingParser()
    args = parser.parse_args()
    TrainingParser.validate_args(args)
    return args


class TrainingParser(argparse.ArgumentParser):

    def __init__(self, **kwargs):
        super(TrainingParser, self).__init__(**kwargs)
        self.add_argument(
            "--cfg",
            help="Path to the config file",
            default="/cs/labs/dina/seanco/xl_mlp_nn/configs/predict_config.yaml",
            type=str,
        )
        self.add_argument(
            "--xl",
            help="Path to Cross Links file in format: residue_number_a chain_id_a residue_number_b "
                 "chain_id_b min_distance max_distance",
            # default='None',
            # default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/xlinks.txt',
            default='/cs/labs/dina/seanco/xl_mlp_nn/predictions/pdb3h32.txt',
            # default='/cs/labs/dina/seanco/Tric/xl/parsed_xl_32.txt',
            # default='/cs/labs/dina/seanco/xl_neighbors/xl_files/P75506.txt',
            type=str,
        )
        self.add_argument(
            "--original_xl",
            help="Path to Cross Links file in format: residue_number_a chain_id_a residue_number_b "
                 "chain_id_b min_distance max_distance. Given only if this file needs update to dimeric form",
            default='/sci/labs/dina/seanco/xl_neighbors/xl_files/pdb3h32.txt',
            type=str,
        )
        self.add_argument(
            "--chain_dimer_dict",
            help="pairs of dimeric chain id. for examle: AD BF CE ",
            default="AD BE CF",
            type=str,
        )
        self.add_argument(
            "--pdb",
            help="Path to pdb files",
            # default=['/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/A.pdb',
            #          '/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/B.pdb'],
            default=['/cs/labs/dina/seanco/xl_parser/pdbs/pdb3h32_E.ent',
                     '/cs/labs/dina/seanco/xl_parser/pdbs/pdb3h32_F.ent'],
            # default=['/cs/labs/dina/seanco/Tric/input/A.pdb',
            #          '/cs/labs/dina/seanco/Tric/input/B.pdb',
            #          '/cs/labs/dina/seanco/Tric/input/C.pdb',
            #          '/cs/labs/dina/seanco/Tric/input/D.pdb',
            #          '/cs/labs/dina/seanco/Tric/input/E.pdb',
            #          '/cs/labs/dina/seanco/Tric/input/F.pdb',
            #          '/cs/labs/dina/seanco/Tric/input/G.pdb',
            #          '/cs/labs/dina/seanco/Tric/input/H.pdb'],
            # default='/cs/labs/dina/seanco/xl_parser/pdbs/alpha_fold/pdb_files/P75506.pdb',
            nargs='+'
        )

        self.add_argument(
            "--solution_pdb",
            help="(optional) Path to solution pdb of multichain protein",
            default='/cs/labs/dina/seanco/xl_parser/pdbs/pdb3h32.ent',
            # default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/A_B.pdb',
            # default='/cs/labs/dina/seanco/Tric/tric_align/debug_res.pdb',
            type=str
        )

        self.add_argument(
            "--out_path",
            help="Path to prediction output file",
            default='/cs/labs/dina/seanco/xl_mlp_nn/predictions/pred_xlinks_3h32.txt',
            # default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/fake_prediction.txt',
            # default='/cs/labs/dina/seanco/Tric/input/fake_prediction.txt',
            type=str
        )

        self.add_argument(
            "--linker",
            help="Linker type. one of: DSSO, DSS, BDP_NHP UNKNOWN LEIKER",
            default='DSSO',
            # default='DSS',
            type=str
        )

        self.add_argument(
            "--multichain",
            help="Is the protein single or multi chain",
            default=True,
            type=bool
        )

    @staticmethod
    def validate_args(args):
        if args.cfg is None or not os.path.isfile(args.cfg):
            raise argparse.ArgumentTypeError(f"Invalid config file path: {args.cfg}")
        if args.linker not in cross_link.LINKER_DICT:
            raise argparse.ArgumentTypeError(f"Invalid linker type: {args.linker}")

    def parse_args(self, args=None, namespace=None):
        """ Parse the input arguments """
        args = super(TrainingParser, self).parse_args(args, namespace)
        return args


def run_prediction(cfg_path, out_path, xl_file, pdb_files, linker, multichain, solution_pdb, original_xl=None, dimers=None):
    train.seed_everything()
    out_file = None
    cfg = train.load_config(None, cfg_path)
    if cfg['output_to_file']:
        out_file = open(out_path, 'w')
    if xl_file != 'None':
        if original_xl is not None and dimers is not None:
            preprocess_xlink_file(original_xl, xl_file, dimers)
        data_name, xl_objects, dimer_dict = pre_process_data(xl_file, pdb_files, linker, solution_pdb, multichain)
        cfg['dataset'] = data_name
        if cfg['predict_type'] == 'real':
            predict(cfg, xl_objects, out_file=out_file, dimer_dict=dimer_dict)
        else:
            fake_predict(cfg, None, xl_objects, out_file=out_file, dimer_dict=dimer_dict)
    else:
        fake_predict(cfg, pdb_files[0], None, out_file=out_file)


def main():
    _args = parse_args()
    run_prediction(_args.cfg, _args.out_path, _args.xl, _args.pdb, _args.linker, _args.multichain, _args.solution_pdb, _args.original_xl, _args.chain_dimer_dict)


if __name__ == "__main__":
    main()

