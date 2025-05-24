CUDA_VISIBLE_DEVICES=1 whisper /home/galimzyanov/temp/audio/1.mp3 \
--output_dir /home/galimzyanov/temp/audio/ \
--model turbo \
--language ru \
--output_format srt \
--word_timestamps True \
--max_line_count 2 \
--max_words_per_line 15 \
--verbose False