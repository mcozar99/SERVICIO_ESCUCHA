import tweepy
import re
import wget
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

# authorization tokens
consumer_key = "mag4RQB6jXTbSydTeHkikxKTD"
consumer_secret = "8RumMMZXZVbi9in9YrQdn61FqMU90S0RzhaFhj6zLbxOzRjqnv"
access_key = "1332295380638310400-QSZkmQdGTcho6Ohx6ouydr1yrxrFfn"
access_secret = "RgS8RDPVYmCVQbd4ooGGCMWAyrYJOaW9QrmmNjFOZLoSI"

# StreamListener class inherits from tweepy.StreamListener and overrides on_status/on_error methods.
class StreamListener(tweepy.StreamListener):
    def on_status(self, status):
        # para probar el algoritmo completo borrar la primera sentencia que restringe lenguajes y rt
        # remove tweets which are not in spanish or retweets
        if status.lang != 'es' or status.retweeted:
            return

        print(status.id_str)
        # if "retweeted_status" attribute exists, flag this tweet as a retweet.
        is_retweet = hasattr(status, "retweeted_status")

        # check if text has been truncated

        if hasattr(status,"extended_tweet"):
            text = status.extended_tweet["full_text"]
        else:
            text = status.text

        # check if this is a quote tweet.
        is_quote = hasattr(status, "quoted_status")
        quoted_text = ""
        if is_quote:
            # check if quoted tweet's text has been truncated before recording it
            if hasattr(status.quoted_status,"extended_tweet"):
                quoted_text = status.quoted_status.extended_tweet["full_text"]
            else:
                quoted_text = status.quoted_status.text

        # remove characters that might cause problems with csv encoding
        text = re.split('[, \n]', text)
        text = ' '.join(text)

        # Para descargarse las fotos de los tweets que se van recopilando, descomentar para usar
        """
        media_files = set()
        media = status.entities.get('media', [])
        if (len(media) > 0):
            media_files.add(media[0]['media_url'])
        for media_file in media_files:
            wget.download(media_file, 'M:/Universidad/4º/CDPS/LABS/p4_creativa/media')
        """

        with open("./corpus/TEST_1.txt", "a", encoding='utf-8') as f:
            #f.write("%s,%s,%s,%s,%s,%s\n" % (status.created_at,status.user.screen_name,is_retweet,is_quote,text,quoted_text))
            f.write("%s\t%s\t%s\n" %(status.id_str, status.user.screen_name, text))

    def on_error(self, status_code):
        print("Encountered streaming error (", status_code, ")")
        sys.exit()

if __name__ == "__main__":
    # complete authorization and initialize API endpoint
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize stream
    streamListener = StreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=streamListener,tweet_mode='extended')
    #with open("out.csv", "w", encoding='utf-8') as f:
        #f.write("date,user,is_retweet,is_quote,text,quoted_text\n")
        #f.write("date, id, user, text\n")
    tags = ['Maradona', 'Messi', 'Hernandez', 'Sevilla', 'Barcelona', 'Papu', 'Cortina', 'Alcalá la Real']
    stream.filter(track=tags)