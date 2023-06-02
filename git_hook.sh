#!/bin/bash

branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')

if [ "$branch" = "development" ]; then
  best_file=""
  best_value=-1
  folder="/home/appuser/de2-final-project/models/"

  for f in "$folder"/*.py; do
      if [ -f "$f" ]; then
          output=$(python3 "$f")
          if [ "$(bc <<< "$output > $best_value")" -eq 1 ]; then
              best_value="$output"
            # Extract the base name without extension and add .joblib
              base_name=$(basename "$f" .py)
              best_file="${folder}${base_name}.joblib"
          fi
      fi
  done
# Adding best model file to Git index
  git add "$best_file"
  git commit -m "Best model committed with an accuracy of $best_value, the filename was $best_file" --no-verify

  git checkout main
  git checkout development -- "$best_file" # checkout only the best model file to main branch
  git checkout development -- "scaler.joblib"
  git checkout development -- "imputer.joblib"
  # Add and commit the best model file in the 'main' branch
  git add "$best_file"
  git commit -m "Merged best model file from development branch: $best_file"
  git push
  echo "Updating Production Server"
  ssh -i /home/appuser/group8.pem appuser@192.168.2.15 <<EOF
  cd de2-final-project
  git checkout main
  git pull
  echo "Pulled changes to production server"
  cd models
  mv "$best_file" best_model.joblib
  echo "Moving file to production server"
  sudo mv best_model.joblib ../production_server
EOF
fi
