echo "downloading finetune_regression_test.zip"
filename='finetune_regression_test.zip'
fileid='1ctx0JIPiOakFsdHIqukPBUxVtcXWB1hH'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
unzip ${filename}
rm ${filename}
mv regression_test/* ./
rm -rf regression_test