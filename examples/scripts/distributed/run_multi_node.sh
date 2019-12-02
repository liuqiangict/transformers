WORK_DIR=/home/qiangliu/Git/Mine/transformers/examples/scripts

#Script for doing all the setup for the app such as installing dependencies
SETUPSCRIPT=${WORK_DIR}/distributed/node_environment_setup.sh

#Runs on each node to setup distributed Python execution 
JOBSCRIPT=${WORK_DIR}/distributed/run_on_node.sh

#Training script is executed on each node using job script
APPSCRIPT=${WORK_DIR}/distributed/app_scripts/multi_target.sh

SETUPLOG=/tmp/DistributedSetupLog
RANKSETUPLOG=/tmp/RankSetupLog
RANKERRORLOG=/tmp/RankErrorLog
JOBLOG=/tmp/DistributedJobLogs
JOBERRORLOG=/tmp/DistributedErrorLogs

#Install Parallel SSH
#echo "Setting up Parallel SSH"
#sudo -H apt-get install pssh

#Run SetupScript on all the nodes
#echo "Running Full Distributed Setup on All Nodes. Check $SETUPLOG"
#parallel-ssh -t 0 -o $SETUPLOG -h ~/mpi-hosts bash $SETUPSCRIPT

#Find appropriate rank for each node
echo "Setting up rank id. Check $RANKSETUPLOG"
#hostname=`hostname -I`
#hostip=`echo $hostname | awk '{print $1}'`

cat /root/.ssh/config | grep worker- | cut -d' ' -f2 > ~/mpi-hosts

master_ip=`head -1 ~/mpi-hosts`
count=1
sudo -H rm /tmp/host-ranks-master
for host in `cat ~/mpi-hosts`
do
    #ip=`grep $host /etc/hosts | awk '{print $1}'`
    if [ "$host" == "$master_ip" ]; then
        rank=0
    else
        rank=$count
        count=$((count+1))
    fi
    ip=`grep -A1 "\<${host}\>" /root/.ssh/config | grep HostName | awk '{print $2}'`
    echo "$ip $host $rank" >> /tmp/host-ranks-master
    echo "$ip $host $rank"
done

echo "/tmp/host-ranks-master:"
cat /tmp/host-ranks-master

parallel-scp -o $RANKSETUPLOG -e $RANKERRORLOG -h ~/mpi-hosts /tmp/host-ranks-master /tmp/ip-ranks

#Run the actual job script
echo "Running Job Script. Check $JOBLOG and $JOBERRORLOG"
parallel-ssh -t 0 -o $JOBLOG -e $JOBERRORLOG -h ~/mpi-hosts bash $JOBSCRIPT $APPSCRIPT

