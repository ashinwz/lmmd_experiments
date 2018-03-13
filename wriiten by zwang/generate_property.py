# Written by Zengrui Wu, LMMD, ECUST, 2016-5-6
# Updated by Zengrui Wu, LMMD, ECUST, 2016-6-23, 2016-6-24, 2016-10-9, 2016-10-29
# Usage: generate_property.py input_file output_file proc_n
import sys, subprocess, fileinput, pybel

def run(command_a, path_in, command_b, path_out):
    assert len(path_in) == len(path_out)
    proc, n = [], len(path_in)
    for i in range(n):
        proc.append(subprocess.Popen(tuple(['/usr/bin/env']) + command_a + tuple([path_in[i]]) + command_b + tuple([path_out[i]])))
    for i in range(n):
        proc[i].wait()
        sys.stdout.write('# ' + path_out[i] + '\n')
        sys.stdout.flush()

proc_n   = int(sys.argv[3])
path_in  = sys.argv[1]
path_out = sys.argv[2]
path_splitted_smi    = tuple(['_0_' + path_in.split('.')[0] + '_splitted_'    + str(i) + '.smi' for i in range(proc_n)])
path_original_mae    = tuple(['_1_' + path_in.split('.')[0] + '_original_'    + str(i) + '.mae' for i in range(proc_n)])
path_applied_mae     = tuple(['_2_' + path_in.split('.')[0] + '_applied_'     + str(i) + '.mae' for i in range(proc_n)])
path_desalted_mae    = tuple(['_3_' + path_in.split('.')[0] + '_desalted_'    + str(i) + '.mae' for i in range(proc_n)])
path_neutralized_mae = tuple(['_4_' + path_in.split('.')[0] + '_neutralized_' + str(i) + '.mae' for i in range(proc_n)])
path_epik_mae        = tuple(['_5_' + path_in.split('.')[0] + '_epik_'        + str(i) + '.mae' for i in range(proc_n)])
path_output_smi      = tuple(['_6_' + path_in.split('.')[0] + '_output_'      + str(i) + '.smi' for i in range(proc_n)])

smiles = [x.split('\t')[1].strip() for x in open(path_in, 'r').readlines()]
raw_data = []
for i, x in enumerate(smiles):
    if '=[C@]=' not in x and '=[C@@]=' not in x: # VERY IMPORTANT!
        x = x.replace('\\\\', '\\')              # VERY IMPORTANT!
        raw_data.append((len(x), x, i))
raw_data.sort()
sys.stdout.write('# Input: ' + str(len(raw_data)) + ' / ' + str(len(smiles)) + '\n')
sys.stdout.flush()

data = [[] for i in range(proc_n)]
for i, x in enumerate(raw_data):
    data[i % proc_n].append(x[1] + '\t' + str(x[2]) + '\n')
for i in range(proc_n):
    open(path_splitted_smi[i], 'wb').writelines(data[i])
    sys.stdout.write('# ' + path_splitted_smi[i] + ': ' + str(len(data[i])) + '\n')
    sys.stdout.flush()

run(tuple(['smiles_to_mae']), path_splitted_smi, tuple(), path_original_mae)
run(tuple(['applyhtreat']),   path_original_mae, tuple(), path_applied_mae)
run(tuple(['desalter']),      path_applied_mae,  tuple(), path_desalted_mae)
run(tuple(['neutralizer']),   path_desalted_mae, tuple(), path_neutralized_mae)
run(tuple(['epik', '-NOJOBID', '-best_neutral', '-ph', '7.0', '-imae']), path_neutralized_mae, tuple(['-omae']), path_epik_mae)
run(tuple(['structconvert', '-imae', '-osmi']), path_epik_mae, tuple(), path_output_smi)

output_data = []
for i in range(proc_n):
    for line in fileinput.input(path_output_smi[i]):
        try:
            mol = pybel.readstring('smi', line.rstrip('\r\n'))
            mol.OBMol.ConvertDativeBonds()
            can_smiles, smiles_index = [x.strip() for x in mol.write('can').split('\t')]
            old_smiles = smiles[int(smiles_index)]
            desc = mol.calcdesc(['MW', 'logP'])
            c_n  = [x.atomicnum for x in mol.atoms].count(6)
            output_data.append((old_smiles, can_smiles,
                '%g' % desc['MW'], '%g' % desc['logP'], '%u' % c_n))
        except:
            sys.stdout.write('ERROR: Failed to convert SMILES "' + line.split('\t')[0] + '"\n')
            sys.stdout.flush()
sys.stdout.write('# Output: ' + str(len(output_data)) + ' / ' + str(len(smiles)) + '\n')
sys.stdout.flush()

f = open(path_out, 'wb')
f.write('\t'.join(['SMILES', 'Canonical SMILES (Desalted)', 'MW', 'LogP', 'nC']) + '\n')
for x in sorted(output_data): f.write('\t'.join(x) + '\n')
f.close()
