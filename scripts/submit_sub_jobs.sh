qsub -J 1-8 run_gulp_sweep.sh
# wait 2s
sleep 2
qsub -J 9-16 run_gulp_sweep.sh
sleep 2
qsub -J 17-24 run_gulp_sweep.sh
sleep 2
qsub -J 25-32 run_gulp_sweep.sh
sleep 2
qsub -J 33-40 run_gulp_sweep.sh
sleep 2
qsub -J 41-48 run_gulp_sweep.sh
sleep 2
qsub -J 49-56 run_gulp_sweep.sh
sleep 2
qsub -J 57-64 run_gulp_sweep.sh
sleep 2
qsub -J 65-72 run_gulp_sweep.sh
sleep 2
qsub -J 73-81 run_gulp_sweep.sh

echo "All jobs submitted."