import json
import logging
import pickle
import string

import nltk
import numpy as np
import pandas as pd
from telegram.ext import Updater, CommandHandler

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def find(bot, update):
    def load_obj(name):
        with open('data/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    stemmer = nltk.stem.porter.PorterStemmer()

    def stem_tokens(tokens):
        return [stemmer.stem(item) for item in tokens]

    def preprocess(text):
        return ' '.join(stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map))))

    def texts_overlap(t1, t2):
        """
            t1 - skill
            t2 - search text
        """

        if pd.isnull(t1) or len(t1) == 0:
            return 0
        if pd.isnull(t2) or len(t2) == 0:
            return 0

        word_count = 0
        for word1 in t1.split(' '):
            if word1 in t2.split(' '):
                word_count += 1

        return word_count / len(t1.split(' '))

    # load stuff
    idfs = load_obj('idfs')
    idfs_mean = np.mean(list(idfs.values()))
    resume_df = pd.read_csv('data/resume_df.csv', sep=';', encoding='cp1251')
    resume_df.skill_set = resume_df.skill_set.map(lambda x: x.strip("[]").replace('\'', '').split(', '))
    resume_df.skill_set_preprocessed = resume_df.skill_set_preprocessed.map(
        lambda x: x.strip("[]").replace('\'', '').split(', '))

    line = update.message.text
    print(line)
    # position = line.split("\", \"")[1].strip('" ')
    # reg = line.split("\", \"")[2].strip('" ')
    desc = line[line.find(' '):]

    # calc skills
    test_df = pd.DataFrame()
    count = 0

    test_df.loc[count, 'text'] = desc
    desc = preprocess(desc)

    skills = []
    for index, row in resume_df.iterrows():
        skill_set = row.skill_set_preprocessed

        for skill in skill_set:
            if skill in desc:
                skills.append(skill)
            elif texts_overlap(skill, desc) >= 0.5:
                skills.append(skill)

    if len(skills) != 0:
        skills = set(skills)
        skills = [x for x in skills if len(x) > 0]
        test_df.loc[count, 'essential_skills'] = str([x for x in skills if idfs.get(x) > idfs_mean])
        test_df.loc[count, 'essential_skills_c'] = len([x for x in skills if idfs.get(x) > idfs_mean])
        test_df.loc[count, 'nice_to_have_skills'] = str([x for x in skills if idfs.get(x) <= idfs_mean])
        test_df.loc[count, 'nice_to_have_skills_c'] = len([x for x in skills if idfs.get(x) <= idfs_mean])

    count += 1

    # result
    result = pd.DataFrame(
        columns=np.append(test_df.columns,
                          np.append(resume_df.columns, ['es_skills_counter', 'nth_skills_counter'])))

    count = 0
    for index, desc in test_df.iterrows():

        for jindex, row in resume_df.iterrows():

            es_skills = desc.essential_skills
            es_skills_counter = 0

            nth_skills = desc.nice_to_have_skills
            nth_skills_counter = 0

            for skill in row.skill_set_preprocessed:
                if pd.notnull(es_skills) and skill in es_skills:
                    es_skills_counter += 1

                if pd.notnull(nth_skills) and skill in nth_skills:
                    nth_skills_counter += 1

            if es_skills_counter > 0 or nth_skills_counter > 0:
                result.loc[count, test_df.columns.values] = desc
                result.loc[count, resume_df.columns.values] = row
                result.loc[count, 'es_skills_counter'] = es_skills_counter
                result.loc[count, 'nth_skills_counter'] = nth_skills_counter

                count += 1

    # build result
    test_df.essential_skills = test_df.essential_skills.map(lambda x: x.strip("[]").replace('\'', '').split(', '))
    test_df.nice_to_have_skills = test_df.nice_to_have_skills.map(
        lambda x: x.strip("[]").replace('\'', '').split(', '))

    j_str = "{"

    # add skills freq

    j_str = j_str + """"es_skills":{0},
                   "es_skills_p":{2},
                   "nth_skills":{1},
                   "nth_skills_p":{3},
                   "resume":[""".format(str([x for x in test_df.essential_skills.values[0]]).replace("'", "\""),
                                        str([x for x in test_df.nice_to_have_skills.values[0]]).replace("'", "\""),
                                        str([idfs.get(x) / np.max(list(idfs.values())) for x in
                                             test_df.essential_skills.values[0]]).replace("'", "\""),
                                        str([idfs.get(x) / np.max(list(idfs.values())) for x in
                                             test_df.nice_to_have_skills.values[0]]).replace("'", "\""))

    for index, row in result.sort_values(['es_skills_counter', 'nth_skills_counter'], ascending=False).iterrows():

        id_url = row.id
        e_s_c = test_df.essential_skills_c.values[0]
        n_s_c = test_df.nice_to_have_skills_c.values[0]

        if e_s_c == 0:
            es_skill_p = 0
        else:
            es_skill_p = (row.es_skills_counter / e_s_c) * 100

        if n_s_c == 0:
            nth_skill_p = 0
        else:
            nth_skill_p = (row.nth_skills_counter / n_s_c) * 100

        j_str = j_str + "{"
        j_str = j_str + """"id":"{0}", "es_skills_p":{1}, "nth_skills":{2}""".format(id_url, int(es_skill_p),
                                                                                     int(nth_skill_p))
        j_str = j_str + "},"

    j_str = j_str.rstrip(',') + ']}'
    j = json.loads(j_str, encoding="cp1251")

    with open('result.json', 'w') as outfile:
        json.dump(j, outfile, ensure_ascii=False)

    bot.send_message(chat_id=update.message.chat_id,
                     text="Ключевые навыки:{0}".format(test_df.essential_skills.values[0]))

    bot.send_message(chat_id=update.message.chat_id,
                     text="Дополнительные навыки:{0}".format(test_df.nice_to_have_skills.values[0]))

    for id in result.id.values[:3]:
        bot.send_message(chat_id=update.message.chat_id,
                         text="https://hh.ru/resume/{0}".format(id))


def main():
    """Start the bot."""
    # Create the EventHandler and pass it your bot's token.
    updater = Updater("487836253:AAEQRmd6SaK3U22XYnsFzPWqLJEAnBACHRY")

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    # dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("find", find))
    # dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    # dp.add_handler(MessageHandler(Filters.text, echo))

    # log all errors
    # dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
