#!/bin/bash
#Python=`which python`

input_data=hdfs://ns3-backup/user/ads_fst/guojing7/user_ad/test
validate_data=hdfs://ns3-backup/user/ads_fst/guojing7/user_ad/test

files=$(hdfs dfs -ls ${input_data}/ | awk '{print $8;}'| tr '\n' ' ')
echo $files

cnt=0
for k in $files; do
{
    echo "hdfs dfs -cat $k | awk -F"," '{if(NF!=32) print 0;}'"
    ret=$(hdfs dfs -cat $k | awk -F"," '{if(NF!=32) print $0}')
    echo "result:[$k][$ret]"
    if [[ $ret -ne "" ]]; then
        echo $k >> checkresult.txt
        #echo "$model"
    fi
} &
    let cnt+=1
    if [[ $cnt -gt 50 ]]; then
        wait
        let cnt=0
    fi
done

files=$(hdfs dfs -ls ${validate_data}/ | awk '{print $8;}'| tr '\n' ' ')
echo $files
cnt=0
for k in $files; do
{
    echo "hdfs dfs -cat $k | awk -F"," '{if(NF!=32) print 0;}'"
    ret=$(hdfs dfs -cat $k | awk -F"," '{if(NF!=32) print $0}')
    echo "result:[$k][$ret]"
    if [[ $ret -ne "" ]]; then
        echo $k >> checkresult.txt
        #echo "$model"
    fi
} &
    let cnt+=1
    if [[ $cnt -gt 50 ]]; then
        wait
        let cnt=0
    fi
done

wait

