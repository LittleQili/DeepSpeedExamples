# bash /localdata_ssd/yjdiao/remote/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts/opt/single_node/run_1.3b.sh \
#  /localdata_ssd/yjdiao/remote/DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b 


# python e2e_rlhf.py --actor-model facebook/opt-1.3b \
#  --reward-model facebook/opt-350m \
#  --deployment-type \
#  --step 1

# nsys profile -o dpspd_train --trace=cuda,nvtx,cudnn,cublas  --gpuctxsw=true -f true \
#   --wait primary \
python e2e_rlhf.py --actor-model facebook/opt-1.3b \
    --reward-model facebook/opt-350m \
    --deployment-type single_node \
    --step 1


nsys launch --trace=cuda,nvtx python e2e_rlhf.py --actor-model facebook/opt-1.3b \
    --reward-model facebook/opt-350m \
    --deployment-type single_node \
    --step 1 &
nsys start
nsys stop


nsys stats -r kernexectrace -f csv -o . --timeunit msec --force-overwrite true 1card_0.nsys-rep
nsys stats -r kernexectrace -f csv -o . --timeunit msec --force-overwrite true 2card_0.nsys-rep
nsys stats -r kernexectrace -f csv -o . --timeunit msec --force-overwrite true 4card_0.nsys-rep
nsys stats -r kernexectrace -f csv -o . --timeunit msec --force-overwrite true 8card_0.nsys-rep
