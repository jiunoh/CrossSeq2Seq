#! /bin/bash

# Code taken from (modified):
# https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt17-multilingual.sh
# MIT licensed. "Copyright (c) Facebook, Inc. and its affiliates. All rights reserved."

SRCS=(
    "ro"
)
TGT=it


ORIG=iwslt17_orig
DATA=data
mkdir -p "$ORIG" "$DATA"

URLS=(
    "https://wit3.fbk.eu/archive/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz"
)
ARCHIVES=(
    "DeEnItNlRo-DeEnItNlRo.tgz"
)

UNARCHIVED_NAME="DeEnItNlRo-DeEnItNlRo"

VALID_SETS=(
    "IWSLT17.TED.dev2010.ro-it"
    "IWSLT17.TED.dev2010.ro-it"
)

TEST_FILE="IWSLT17.TED.tst2010"

TEST_PAIRS=(
    "ro it"
)

echo "pre-processing train data..."
for SRC in "${SRCS[@]}"; do
    for LANG in "${SRC}" "${TGT}"; do
        cat "train.tags.${SRC}-${TGT}.${LANG}" \
            | grep -v '<url>' \
            | grep -v '<talkid>' \
            | grep -v '<keywords>' \
            | grep -v '<speaker>' \
            | grep -v '<reviewer' \
            | grep -v '<translator' \
            | grep -v '<doc' \
            | grep -v '</doc>' \
            | sed -e 's/<title>//g' \
            | sed -e 's/<\/title>//g' \
            | sed -e 's/<description>//g' \
            | sed -e 's/<\/description>//g' \
            | sed 's/^\s*//g' \
            | sed 's/\s*$//g' \
            > "$DATA/train.${SRC}-${TGT}.${LANG}"
    done
done

echo "pre-processing valid data..."
for ((i=0;i<${#SRCS[@]};++i)); do
    SRC=${SRCS[i]}
    VALID_SET=${VALID_SETS[i]}
    for FILE in ${VALID_SET[@]}; do
        for LANG in "$SRC" "$TGT"; do
            grep '<seg id' "${FILE}.${LANG}.xml" \
                | sed -e 's/<seg id="[0-9]*">\s*//g' \
                | sed -e 's/\s*<\/seg>\s*//g' \
                | sed -e "s/\’/\'/g" \
                >> "$DATA/valid.${SRC}-${TGT}.${LANG}"
        done
    done
done

echo "pre-processing test data..."

for TEST_PAIR in "${TEST_PAIRS[@]}"; do
    TEST_PAIR=($TEST_PAIR)
    SRC=${TEST_PAIR[0]}
    TGT=${TEST_PAIR[1]}
    for LANG in "$SRC" "$TGT"; do
        grep '<seg id' "${TEST_FILE}.${SRC}-${TGT}.${LANG}.xml" \
            | sed -e 's/<seg id="[0-9]*">\s*//g' \
            | sed -e 's/\s*<\/seg>\s*//g' \
            | sed -e "s/\’/\'/g" \
            > "$DATA/test.${SRC}-${TGT}.${LANG}"
    done
done

