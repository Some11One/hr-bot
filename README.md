# hr-bot

Simple telegram bot. It finds resume on hh.ru based on text. Launch python-bot.py, go to checkgroundBot in telegram and type /find + your text to search for a suitable resume. 

It only supports 30 resume database for now, so to be a complete project you will need to increase /data/resume_df.csv file.

Algorithm to find resumes: nlp preproccessing + skills matching. So the bot is trying to get neccessary skills from your text and find resume with most similar skills. 
