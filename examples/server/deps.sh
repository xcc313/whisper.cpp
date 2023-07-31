#!/bin/bash
# Download and update deps for binary

# get the directory of this script file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PUBLIC=$DIR/public


FILES=$(ls $PUBLIC)

for FILE in $FILES; do
  func=$(echo $FILE | tr '.' '_')
  echo "generate $FILE.hpp ($func)"
  # xxd -n $func -i $PUBLIC/$FILE > $DIR/$FILE.hpp
  xxd -i $PUBLIC/$FILE > $DIR/$FILE.hpp
done
