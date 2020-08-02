import datetime
import argparse
import bart_sum
import logging
import presumm.presumm as presumm

logger = logging.getLogger(__name__)

def do_summarize(contents):
    document = str(contents)
    logger.info("Document Created")


    doc_length = len(document.split())
    logger.info("Document Length: " + str(doc_length))

    min_length = int(doc_length/10)#int(doc_length/6)
    logger.info("min_length: " + str(min_length))
    max_length = min_length+2000
    logger.info("max_length: " + str(max_length))

    transcript_summarized = summarizer.summarize_string(document, min_length=min_length, max_length=max_length)
    with open("summarized.txt", 'a+') as file:
        file.write("\n" + str(datetime.datetime.now()) + ":\n")
        file.write(transcript_summarized + "\n")

parser = argparse.ArgumentParser(description='Summarization of text using CMD prompt')
parser.add_argument('-m', '--model', choices=["bart", "presumm"], required=True,
                    help='machine learning model choice')
parser.add_argument('--bart_checkpoint', default=None, type=str, metavar='PATH',
                    help='[BART Only] Path to optional checkpoint. Semsim is better model but will use more memory and is an additional 5GB download. (default: none, recommended: semsim)')
parser.add_argument('--bart_state_dict_key', default='model', type=str, metavar='PATH',
                    help='[BART Only] model state_dict key to load from pickle file specified with --bart_checkpoint (default: "model")')
parser.add_argument('--bart_fairseq', action='store_true',
                    help='[BART Only] Use fairseq model from torch hub instead of huggingface transformers library models. Can not use --bart_checkpoint if this option is supplied.')
parser.add_argument('--text', default=None, type=str,
                    help='Optional text to summarize if you cannot paste it using an interactive shell.')
parser.add_argument("-l", "--log", dest="logLevel", default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help="Set the logging level (default: 'Info').")
args = parser.parse_args()

logging.basicConfig(format="%(asctime)s|%(name)s|%(levelname)s> %(message)s", level=logging.getLevelName(args.logLevel))

logger.info("Loading Model")
if args.model == "bart":
    summarizer = bart_sum.BartSumSummarizer(checkpoint=args.bart_checkpoint,
                                            state_dict_key=args.bart_state_dict_key,
                                            hg_transformers=(not args.bart_fairseq))
elif args.model == "presumm":
    summarizer = presumm.PreSummSummarizer()

txt = '''As the majority of adults spend most of their time at work, being content with your career is a crucial part of a person’s health and happiness. This essay will first suggest fair pay as a key element leading to job satisfaction and it will then state that it is not very likely that everyone can be happy with their job.

The most important thing that leads to someone being satisfied at work is being compensated fairly. If those more senior than you respect you as a person and the job you are doing then you feel like you are valued. A fair salary and benefits are important marks of respect and if you feel you are being underpaid you will either resent your bosses or look for another job. These two factors came top of a recent job satisfaction survey conducted by Monster.com, that found that 72% of people were pleased with their current role if their superiors regularly told them they were appreciated.

With regards to the question of happiness for all workers, I think this is and always will be highly unlikely. The vast majority of people fail to reach their goals and end up working in a post they don’t really care about in return for a salary. This money is just enough to pay their living expenses which often means they are trapped in a cycle of disenchantment. For example, The Times recently reported that 89% of office workers would leave their jobs if they did not need the money.

In conclusion, being satisfied with your trade or profession is an important part of one’s well-being and respect from one’s colleagues and fair pay can improve your level of happiness, however, job satisfaction of all workers is an unrealistic prospect.'''

random_txt='''As the majority of adults spend most of their time at work, being content with your career is a crucial part of a person’s health and happiness. This essay will first suggest fair pay as a key element leading to job satisfaction and it will then state that it is not very likely that everyone can be happy with their job.

It is irrefutable that cinema is a source of entertainment for the young and the adults of the nation. Comedy movies lighten the mood and relieves a person from all day stress. Furthermore, traditional scripts were based on the fiction to entertain the spectators. They mostly illustrated comic characters. However, there is a conspicuous change observed in the storylines nowadays. 

There is a colossal upsurge in the movies that create awareness about the social evils of the society. To illustrate, media encourage women empowerment. Additionally, numerous films are made to highlight issues like child labour or domestic violence. Consequently, eradicating such erroneous deeds from the country. Also, various movies are based on original characters from our enriched history. Thus, the students learn a lot more than they see such films. For instance, a famous movie Jodha Akbar narrated the life of last Mughal emperor and revived the long lost ancestors who ruled our country. 

In conclusion, being satisfied with your trade or profession is an important part of one’s well-being and respect from one’s colleagues and fair pay can improve your level of happiness, however, job satisfaction of all workers is an unrealistic prospect.'''
do_summarize(random_txt)
#if args.text:
#do_summarize(args.text)
# else:
#     try:
#         while True:
#             print("Enter/Paste your content. Ctrl-D or Ctrl-Z (windows) to save it. Ctrl-C to exit.")
#             contents = ""
#             while True:
#                 try:
#                     line = input()
#                 except EOFError:
#                     break
#                 contents += (line.strip()+ " ")

#             do_summarize(contents)

#     except KeyboardInterrupt:
#         print("Exiting...")
