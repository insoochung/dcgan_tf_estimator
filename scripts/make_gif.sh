for type in 'random' 'interpolation'
do
  convert -delay 20 -loop 0 ../result/*/$type.jpg ../result/$type.gif
done
