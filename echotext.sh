#!/bin/bash

while true; do echo -n echo $(tail -n 3 TF_MultiClass_NN_INCEPTION-output.txt); echo " - "; date; sleep 1; done
