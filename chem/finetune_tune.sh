#### GIN fine-tuning
runseed=$1
device=$2
split=scaffold

### for GIN
#for dataset in pcba
#for dataset in bace bbbp clintox hiv muv sider tox21 toxcast
for dataset in bace bbbp clintox
do
#for unsup in contextpred infomax edgepred masking
for unsup in contextpred infomax edgepred masking
do

model_file=${unsup}
python -c "print('unsup: ' + '$model_file')"
python -c "print('dataset: ' + '$dataset')"
python -c "print('check: 1')"
python finetune.py --input_model_file model_gin/${model_file}.pth --split $split --filename ${dataset}/gin_${model_file} --device $device --runseed $runseed --gnn_type gin --dataset $dataset

model_file=supervised_${unsup}
python -c "print('unsup: ' + '$model_file')"
python -c "print('dataset: ' + '$dataset')"
python -c "print('check: 2')"
python finetune.py --input_model_file model_gin/${model_file}.pth --split $split --filename ${dataset}/gin_${model_file} --device $device --runseed $runseed --gnn_type gin --dataset $dataset
done

python -c "print('check: 3')"
python finetune.py --split $split --filename ${dataset}/gin_nopretrain --device $device --runseed $runseed --gnn_type gin --dataset $dataset
#python -c "print('check: 4')"
#python finetune.py --split $split --input_model_file model_gin/supervised.pth --filename ${dataset}/gin_supervised --device $device --runseed $runseed --gnn_type gin --dataset $dataset


### for other GNNs
#for gnn_type in gcn gat graphsage
#do
#python finetune.py --split $split --filename ${dataset}/${gnn_type}_nopretrain --device $device --runseed $runseed --gnn_type $gnn_type --dataset $dataset

#model_file=${gnn_type}_supervised_contextpred
#python finetune.py --input_model_file model_architecture/${model_file}.pth --split $split --filename ${dataset}/${model_file} --device $device --runseed $runseed --gnn_type $gnn_type --dataset $dataset

#done
done


