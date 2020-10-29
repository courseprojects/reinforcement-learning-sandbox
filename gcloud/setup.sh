sudo gcsfuse --implicit-dirs --dir-mode "777"  \
-o allow_other -o nonempty gs://robot-license ./
