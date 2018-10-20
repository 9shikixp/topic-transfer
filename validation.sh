cd ~/work/topic-transfer/src/
case $HOSTNAME in
  "clover11" ) DATA=yelpf;;
  "clover12" ) DATA=db_pedia ;;
  "denso" ) DATA=ag_news ;;
esac
echo $DATA
for i in `seq $1 2 4`
do
  echo python topic-train-soft.py -m $DATA -g $1 -d 17 -v $i
  python topic-train-soft.py -m $DATA -g $1 -d 17 -v $i
  echo python transfer-soft.py -m $DATA -g $1 -d 17 -v $i
  python transfer-soft.py -m $DATA -g $1 -d 17 -v $i
done
START=$((($1+1)%2))
for i in `seq $START 2 4`
do
  echo python topic-train-hard.py -m $DATA -g $1 -d 17 -v $i
  python topic-train-hard.py -m $DATA -g $1 -d 17 -v $i
  echo python transfer-hard.py -m $DATA -g $1 -d 17 -v $i
  python transfer-hard.py -m $DATA -g $1 -d 17 -v $i
done
for i in `seq $1 2 4`
do
  echo python train-scratch.py -m $DATA -g $1 -d 17 -v $i
  python train-scratch.py -m $DATA -g $1 -d 17 -v $i
done
