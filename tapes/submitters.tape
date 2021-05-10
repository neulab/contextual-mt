submitter slurm :: COMMANDS TASK_VARIABLES CONFIGURATION {
    action run > exit_code {
        # define script (random suffix for parallel acess)
        suffix=$((1 + $RANDOM % 1024))
        wrapper="slurm_script_${suffix}.sh"
        rm -f $wrapper

        echo "#!/bin/bash" > $wrapper
        echo "set -e # stop on errors" >> $wrapper
        echo "set -o pipefail # stop on pipeline errors" >> $wrapper
        echo "set -u # stop on undeclared variables" >> $wrapper
        echo "set -x # show each command as it is executed" >> $wrapper

        # print actual jobs 
        echo "$COMMANDS" >> $wrapper
        srun --mem=$mem --gres=gpu:$gpus -c $cpus --job-name=$TASK --time=0 bash $wrapper
        rm -f $wrapper
  }
}

submitter cuda_async :: COMMANDS TASK_VARIABLES CONFIGURATION {
    action run > exit_code {
        # define script we are gonna wrong
        wrapper="cuda_script.sh"
        rm -f $wrapper

        echo "#!/bin/bash" > $wrapper
    
        # print actual jobs 
        echo "$COMMANDS" >> $wrapper
        CUDA_VISIBLE_DEVICES=$(get_free_gpu --is-empty) bash $wrapper
    }
}