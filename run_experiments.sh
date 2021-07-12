#!/bin/bash


function TrainSimpleModels {
  aux=$1
  ((aux=aux-1))
  model=${SimpleModelsArray[$aux]}
  printf -v p_model "%02d" $1
  output="./out/TFG/PROBS_$1_$2.txt"
  command="python ./actions.py $1 $2 | tee ${output}"
  #command="python ./actions.py $1 $2"
  echo $command
  eval $command
}

# 1ª posicion N_SAMPLES segunda posicion PERCENT


#TrainSimpleModels 100000 40
#TrainSimpleModels 100000 30
TrainSimpleModels 100000 20
#TrainSimpleModels 100000 10
#TrainSimpleModels 100 40
#TrainSimpleModels 100 30
TrainSimpleModels 100 20
#TrainSimpleModels 100 10
#TrainSimpleModels 50 40
#TrainSimpleModels 50 30
TrainSimpleModels 50 20
#TrainSimpleModels 50 10
#TrainSimpleModels 25 40
#TrainSimpleModels 25 30
TrainSimpleModels 25 20
#TrainSimpleModels 25 10
#TrainSimpleModels 10 40
#TrainSimpleModels 10 30
TrainSimpleModels 10 20
#TrainSimpleModels 10 10
#TrainSimpleModels 5 40
#TrainSimpleModels 5 30
TrainSimpleModels 5 20
#TrainSimpleModels 5 10

