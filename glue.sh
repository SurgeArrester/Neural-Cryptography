for k in 1 2 3 4 5
do
    for n in 1 2 3 4 5
    do
        for l in 1 2 3 4 5
        do
            for synch in 100 200 300 400 500
            do
                mpiexec -n 8 /home/cameron/Dropbox/University/Liverpool/ParallelProgramming/Assignment3/v4/bin/geometricParallel $k $n $l $synch -c
                sleep 1
            done
        done
    done
done
