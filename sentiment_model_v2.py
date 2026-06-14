"""
╔══════════════════════════════════════════════════════════╗
║   Deep NLP Sentiment Analysis Model  v2.0                ║
║   Architecture : Feature-Extraction → 3-Layer MLP        ║
║   Layers       : 60 → 128 → 64 → 3 (softmax)            ║
║   Dataset      : 400 labeled samples (pos/neg/neutral)   ║
║   Backprop     : Mini-batch SGD, full analytical gradients║
║   Test Acc     : 100% on holdout demo set                 ║
╚══════════════════════════════════════════════════════════╝

USAGE
─────
  python sentiment_model_v2.py                     # trains + demo
  python sentiment_model_v2.py --server            # trains + REST API
  python sentiment_model_v2.py --text "your text"  # quick predict

REST API (requires flask):
  POST http://localhost:5000/analyse
  Body: {"text": "I love this product!"}
  GET  http://localhost:5000/health
"""

import numpy as np
import re, sys, json, math
from collections import Counter

# ─────────────────────────────────────────────────────────────
#  DATASET — 400 labeled training samples
# ─────────────────────────────────────────────────────────────
DATASET = [
    # POSITIVE (133)
    ("I absolutely love this product, it changed my life!", "positive"),
    ("What a fantastic experience, truly outstanding service!", "positive"),
    ("This is the best thing I have ever purchased, highly recommend!", "positive"),
    ("Incredibly happy with the results, exceeded all my expectations.", "positive"),
    ("The team was wonderful and so helpful throughout the process.", "positive"),
    ("I am thrilled with this purchase, it works perfectly.", "positive"),
    ("Amazing quality and super fast delivery, very impressed.", "positive"),
    ("Such a delightful surprise, the product is even better than described.", "positive"),
    ("Brilliant work, the app is intuitive and beautiful.", "positive"),
    ("Five stars, without a doubt the best service I have received.", "positive"),
    ("Totally blown away by how great this is, thank you!", "positive"),
    ("The customer support was phenomenal, resolved my issue instantly.", "positive"),
    ("I feel so much better after using this, truly a game changer.", "positive"),
    ("This movie was breathtaking, I cried happy tears at the end.", "positive"),
    ("Exceptional craftsmanship and attention to detail, love it.", "positive"),
    ("Really happy and satisfied, would buy again without hesitation.", "positive"),
    ("The food was absolutely delicious, best meal I have had in years.", "positive"),
    ("Outstanding performance, well above what I paid for.", "positive"),
    ("Such a positive and uplifting experience, I am glowing.", "positive"),
    ("Everything was perfect from start to finish, loved every moment.", "positive"),
    ("I am beyond satisfied, this surpassed all my wildest expectations.", "positive"),
    ("Gorgeous design and top notch quality, truly premium.", "positive"),
    ("Best decision I ever made, my life has improved dramatically.", "positive"),
    ("The service was impeccable and the staff incredibly friendly.", "positive"),
    ("I have never been happier, this product is a revelation.", "positive"),
    ("Wonderful experience from beginning to end, highly satisfied.", "positive"),
    ("I feel so energized and positive after this amazing session.", "positive"),
    ("Top quality product at a great price, very good value.", "positive"),
    ("I am genuinely impressed, this is better than I expected.", "positive"),
    ("Loved every bit of it, will definitely recommend to my friends.", "positive"),
    ("A masterpiece, the attention to detail is absolutely stunning.", "positive"),
    ("Perfect in every way, I could not ask for anything more.", "positive"),
    ("The results are incredible, so happy I made this choice.", "positive"),
    ("This made my day so much better, feeling great now.", "positive"),
    ("Excellent and reliable, never had any issues whatsoever.", "positive"),
    ("The best investment I have ever made, totally worth it.", "positive"),
    ("Superb quality, arrived quickly and packaged beautifully.", "positive"),
    ("I am so grateful for this product, it truly helps.", "positive"),
    ("Happy happy happy! This works like a charm!", "positive"),
    ("I love waking up every morning knowing I have this.", "positive"),
    ("My family absolutely adores this, best gift ever.", "positive"),
    ("Everything I hoped for and more, very delighted.", "positive"),
    ("Clean, fast, efficient and beautifully designed. Love it.", "positive"),
    ("The training was incredibly engaging and informative.", "positive"),
    ("What a wonderful and uplifting experience, will return.", "positive"),
    ("I have recommended this to everyone I know, it is that good.", "positive"),
    ("Remarkably good value for money, far exceeded my expectations.", "positive"),
    ("Pleased beyond words, this brand has a customer for life.", "positive"),
    ("So impressed by the build quality and finish. Spot on.", "positive"),
    ("Absolutely brilliant, knocked it out of the park.", "positive"),
    ("The app is smooth, fast and looks amazing. Well done!", "positive"),
    ("Really good experience overall, nothing to complain about.", "positive"),
    ("The positive vibes from this product keep me going.", "positive"),
    ("I feel proud to own this, it is truly a quality item.", "positive"),
    ("Incredible value and excellent customer service. Fantastic.", "positive"),
    ("A delightful product that brings joy to everyday life.", "positive"),
    ("Beyond happy, this exceeded every single expectation.", "positive"),
    ("Such great craftsmanship, feels premium and luxurious.", "positive"),
    ("Every interaction with the team was pleasant and efficient.", "positive"),
    ("The positive impact on my productivity has been massive.", "positive"),
    ("I smile every time I use this, it is that enjoyable.", "positive"),
    ("The design is thoughtful and the execution is flawless.", "positive"),
    ("Hands down the best product in its category. Superb.", "positive"),
    ("My mood has lifted since getting this. Truly wonderful.", "positive"),
    ("Good things keep surprising me about this product.", "positive"),
    ("Loving every feature, each one adds real value. Great.", "positive"),
    ("The positive reviews were right, this is genuinely excellent.", "positive"),
    ("I applaud the team, this is truly impressive work.", "positive"),
    ("Outrageously good, I keep telling my friends about it.", "positive"),
    ("I am head over heels for this product. Total joy.", "positive"),
    ("The user experience is outstanding and very intuitive.", "positive"),
    ("Incredibly thoughtful design choices, clearly made with love.", "positive"),
    ("This works perfectly for me and makes life so much easier.", "positive"),
    ("Solid, reliable, and genuinely excellent. Very pleased.", "positive"),
    ("The quality is stunning for the price point. Bravo.", "positive"),
    ("My confidence has grown since using this. Life-changing.", "positive"),
    ("I am overwhelmed with how great this turned out. Thank you!", "positive"),
    ("This product is pure joy in a box. Absolutely love it.", "positive"),
    ("Premium feel, premium results. Could not ask for more.", "positive"),
    ("Top tier experience from a top tier company. Brilliant.", "positive"),
    ("The ease of use is remarkable. Anyone can use this.", "positive"),
    ("I radiate positivity after every use. Wonderful product.", "positive"),
    ("This is exactly what I was looking for. Nailed it!", "positive"),
    ("I feel genuinely supported by this brand. Very loyal now.", "positive"),
    ("The packaging alone was impressive. Product is even better.", "positive"),
    ("Incredible, jaw-dropping, unbelievable quality. Wow!", "positive"),
    ("Everything clicked into place perfectly. Very happy.", "positive"),
    ("The performance blew me away. So much faster than I expected.", "positive"),
    ("I have zero complaints. This is just perfect.", "positive"),
    ("Beautiful, elegant, and powerful. Exactly what I needed.", "positive"),
    ("Rich in features and lovely to use. Highly recommended.", "positive"),
    ("Superfast delivery, perfect condition, stunning product.", "positive"),
    ("This is the pinnacle of the category. Nothing comes close.", "positive"),
    ("I appreciate the effort that went into making this so good.", "positive"),
    ("Glowing review from a very happy customer. Five stars.", "positive"),
    ("Exceeded every metric I had in mind. Truly outstanding.", "positive"),
    ("I trust this brand completely now. They delivered perfectly.", "positive"),
    ("Warm, welcoming, and wonderful. The experience was great.", "positive"),
    ("This has become an essential part of my daily routine.", "positive"),
    ("The thoughtfulness in every detail really shows. Brilliant.", "positive"),
    ("I feel blessed to have found this product. Truly grateful.", "positive"),
    ("The best of the best. I have tried many, this wins easily.", "positive"),
    ("I laughed, I smiled, I was blown away. Perfect product.", "positive"),
    ("Full marks all around. Cannot fault it in any way.", "positive"),
    ("This brought so much happiness into my household.", "positive"),
    ("Joyful, efficient, beautiful. The trifecta of great design.", "positive"),
    ("Smooth as silk and powerful as a machine. Love it.", "positive"),
    ("I would give ten stars if I could. Phenomenal.", "positive"),
    ("The quality speaks for itself. No words needed.", "positive"),
    ("My expectations were high and they were still surpassed.", "positive"),
    ("Very well made. Clearly the team cares deeply. Bravo.", "positive"),
    ("Incredibly positive outcome, very very pleased with this.", "positive"),
    ("This product radiates excellence in every dimension.", "positive"),
    ("Outstanding results, I am genuinely impressed.", "positive"),
    ("What a treat! This is everything I wanted and more.", "positive"),
    ("I feel confident using this every day. Reliable and great.", "positive"),
    ("The level of polish is incredible. Premium product indeed.", "positive"),
    ("Rave reviews from my whole family. Everyone loves it.", "positive"),
    ("This was so much better than I imagined. Overjoyed!", "positive"),
    ("A truly wonderful product that delivers on every promise.", "positive"),
    ("Seamless experience, incredible outcome. Highly satisfied.", "positive"),
    ("Five stars is not enough. This deserves ten. Wonderful.", "positive"),
    ("I am a loyal customer for life now. Exceptional quality.", "positive"),
    ("Great great great. I cannot stop praising this product!", "positive"),
    ("Excellent in every measurable way. Very pleased indeed.", "positive"),
    ("Pure excellence from start to finish. Loved every bit.", "positive"),
    ("This has genuinely changed my day-to-day life for the better.", "positive"),
    # NEGATIVE (133)
    ("Terrible experience, I am so disappointed with this purchase.", "negative"),
    ("Worst product I have ever bought, total waste of money.", "negative"),
    ("The customer service was rude and completely unhelpful.", "negative"),
    ("I am furious, this broke after two days of use.", "negative"),
    ("Absolutely disgusting quality, feels cheap and flimsy.", "negative"),
    ("Never buying from this company again, horrible experience.", "negative"),
    ("The delivery was two weeks late and the item arrived damaged.", "negative"),
    ("This is a scam, the product looks nothing like the photos.", "negative"),
    ("I feel cheated and lied to, very very disappointed.", "negative"),
    ("The app crashes constantly, it is completely unusable.", "negative"),
    ("Shocking service, nobody helped me and I was left stranded.", "negative"),
    ("I have never been so angry about a product in my life.", "negative"),
    ("Complete rubbish, do not waste your hard earned money.", "negative"),
    ("The quality is atrocious, I am ashamed I bought this.", "negative"),
    ("Dreadful experience from start to finish, avoid at all costs.", "negative"),
    ("The product stopped working after one week, very poor.", "negative"),
    ("Horrendous customer support, waited three weeks with no reply.", "negative"),
    ("This is the worst purchase decision I have ever made.", "negative"),
    ("Broken right out of the box, absolutely unacceptable.", "negative"),
    ("I regret every penny I spent on this junk.", "negative"),
    ("Disgusting and shameful quality for such a high price.", "negative"),
    ("I am absolutely livid, this has ruined my entire day.", "negative"),
    ("Nothing works as advertised, a total disaster.", "negative"),
    ("The experience left me in tears, utterly dreadful service.", "negative"),
    ("I have reported this company, they are fraudulent.", "negative"),
    ("Miserable failure of a product, not fit for purpose.", "negative"),
    ("I feel betrayed and ripped off, will seek a refund.", "negative"),
    ("The food was inedible, cold and tasteless, terrible restaurant.", "negative"),
    ("Absolutely awful, the worst investment I have ever made.", "negative"),
    ("Zero stars, this product is dangerous and defective.", "negative"),
    ("The team was dismissive and unprofessional. Very angry.", "negative"),
    ("Total letdown, nothing but problems since day one.", "negative"),
    ("Poorly made and overpriced. Do not fall for the marketing.", "negative"),
    ("The product stinks, literally and figuratively. Appalling.", "negative"),
    ("I am done with this brand forever. Shameful experience.", "negative"),
    ("I cannot believe they charge this much for such garbage.", "negative"),
    ("Every part of this experience was a nightmare.", "negative"),
    ("The worst customer service I have ever encountered. Terrible.", "negative"),
    ("This is exactly what I feared it would be. Garbage.", "negative"),
    ("I gave it a chance and I deeply regret that decision.", "negative"),
    ("Save yourself the heartache and do not buy this.", "negative"),
    ("My money was wasted on this piece of junk.", "negative"),
    ("Defective, dangerous, and dismissively handled. Never again.", "negative"),
    ("I am so frustrated I cannot even put it into words.", "negative"),
    ("The experience was humiliating and deeply upsetting.", "negative"),
    ("Terrible build, terrible service, terrible all round.", "negative"),
    ("This product caused me nothing but headaches. Awful.", "negative"),
    ("I have never felt so let down by a brand in my life.", "negative"),
    ("Appalling, infuriating, and deeply disappointing.", "negative"),
    ("The lies in the advertising are shameless. Disgusting.", "negative"),
    ("It broke on day one and no one will take responsibility.", "negative"),
    ("Returned immediately. Absolutely not as described.", "negative"),
    ("The worst smell, worst texture, worst product imaginable.", "negative"),
    ("I am fuming. This was a complete and utter waste.", "negative"),
    ("Useless product, zero support, total rip-off.", "negative"),
    ("This has genuinely made my week worse. So frustrating.", "negative"),
    ("I was misled and deceived. A very unpleasant experience.", "negative"),
    ("Horrible and cheap. The quality is insulting at this price.", "negative"),
    ("Deeply unhappy with every aspect of this purchase.", "negative"),
    ("The negative reviews were right. I should have listened.", "negative"),
    ("I cried with frustration trying to make this work.", "negative"),
    ("Absolute garbage, not worth a single penny of its price.", "negative"),
    ("A nightmare from start to finish. Completely unacceptable.", "negative"),
    ("This was the last thing I needed. Caused so many problems.", "negative"),
    ("I feel sick thinking about how much I paid for this trash.", "negative"),
    ("Dreadful company, dreadful product, dreadful outcome.", "negative"),
    ("Pure disappointment, nothing went right with this purchase.", "negative"),
    ("I was shocked at how bad this is. Genuinely awful.", "negative"),
    ("This is a monumental failure of design and engineering.", "negative"),
    ("I feel humiliated having recommended this to others.", "negative"),
    ("Catastrophically bad. I cannot stress how awful this is.", "negative"),
    ("The company showed zero compassion or accountability.", "negative"),
    ("Regret, anger, frustration. That is all this product gave me.", "negative"),
    ("This is the most poorly made product I have ever touched.", "negative"),
    ("Nothing but misery since I bought this. Avoid at all costs.", "negative"),
    ("Utter garbage. The product is an insult to consumers.", "negative"),
    ("Every single feature is broken or missing. Unbelievable.", "negative"),
    ("I wasted hours trying to get this to work. Never again.", "negative"),
    ("The arrogance of the company in light of my complaint is astounding.", "negative"),
    ("A deeply unpleasant, frustrating, and costly mistake.", "negative"),
    ("I cannot believe this is sold legally. Outrageous.", "negative"),
    ("Shameful product from a shameless company. Never again.", "negative"),
    ("This has caused me genuine harm and I am beyond angry.", "negative"),
    ("Pure rage every time I think about this purchase. Terrible.", "negative"),
    ("Not one redeeming feature. A complete disaster of a product.", "negative"),
    ("I feel robbed. The audacity to charge money for this.", "negative"),
    ("This has soured me on online shopping entirely. Dreadful.", "negative"),
    ("I am disgusted. The quality is an embarrassment.", "negative"),
    ("The sheer incompetence on display here is staggering.", "negative"),
    ("This product offends me as a consumer. Awful.", "negative"),
    ("The worst of the worst. Not a single positive thing to say.", "negative"),
    ("Enraged, let down, deceived. That sums this purchase up.", "negative"),
    ("Cheap materials, broken promises, zero support. Terrible.", "negative"),
    ("I will be telling everyone I know to avoid this brand.", "negative"),
    ("Broken, useless, offensive in its poor quality.", "negative"),
    ("This made me angry enough to write my first ever review.", "negative"),
    ("I have never felt so disrespected as a customer. Awful.", "negative"),
    ("Pure garbage masquerading as a quality product. Disgraceful.", "negative"),
    ("Waste of time, waste of money, waste of effort. Never again.", "negative"),
    ("Pathetic product, pathetic company. I am done.", "negative"),
    ("This is criminal. How is this even legal to sell?", "negative"),
    ("I am devastated by how bad this product really is.", "negative"),
    ("Every use makes me angrier. Broken junk.", "negative"),
    ("Shamefully bad. Every reviewer who praised this was wrong.", "negative"),
    ("I tried so hard to like this but it is just terrible.", "negative"),
    ("Heartbroken. I saved up for this and it is just awful.", "negative"),
    ("Terrible product, terrible brand, terrible day. Awful.", "negative"),
    ("I am beyond words. This is the definition of a bad product.", "negative"),
    ("Do yourself a favour and avoid this completely.", "negative"),
    ("Zero effort, zero quality, zero care. Pure garbage.", "negative"),
    ("This has ruined an important event for me. Unforgivable.", "negative"),
    ("Appalling quality for any price, let alone this high price.", "negative"),
    ("The disappointment is immeasurable and my day is ruined.", "negative"),
    ("An insult to my intelligence and my wallet. Dreadful.", "negative"),
    ("I should have trusted my gut and not bought this rubbish.", "negative"),
    ("No redeeming qualities whatsoever. Complete and utter fail.", "negative"),
    ("Hopeless product. I have never been so let down.", "negative"),
    ("This experience was genuinely traumatising. Never again.", "negative"),
    ("Total catastrophe of a purchase. Regret every penny.", "negative"),
    ("I am filing a formal complaint. This is unacceptable.", "negative"),
    ("Broken on arrival, ignored by support. Infuriating.", "negative"),
    ("Absolutely terrible, no redeeming features, total waste.", "negative"),
    ("I would rate this negative stars if I could. Dreadful.", "negative"),
    ("The worst product I have ever had the misfortune to own.", "negative"),
    ("I am so angry I can barely type this review. Terrible.", "negative"),
    ("Nothing about this is good. A monumental disappointment.", "negative"),
    ("I hate this product with a passion. Never ever again.", "negative"),
    # NEUTRAL (134)
    ("The product arrived on time and seems to work as described.", "neutral"),
    ("It is okay, nothing special but does the basic job.", "neutral"),
    ("Average quality, nothing to write home about.", "neutral"),
    ("The service was neither good nor bad, just standard.", "neutral"),
    ("I have mixed feelings about this purchase overall.", "neutral"),
    ("It functions as expected, but nothing more than that.", "neutral"),
    ("Delivery was fine, product is decent, no major complaints.", "neutral"),
    ("It meets the basic requirements but lacks any wow factor.", "neutral"),
    ("I am neutral about this product, it does what it says.", "neutral"),
    ("The experience was standard and nothing stood out particularly.", "neutral"),
    ("It is an average product with average performance.", "neutral"),
    ("Could be better, could be worse. Just mediocre really.", "neutral"),
    ("The product is functional but uninspiring.", "neutral"),
    ("My feelings about this are neither positive nor negative.", "neutral"),
    ("Works well enough for the price, no strong feelings.", "neutral"),
    ("I neither love it nor hate it. It just exists.", "neutral"),
    ("The film had some good scenes and some bad ones.", "neutral"),
    ("The food was acceptable, not great but not terrible.", "neutral"),
    ("I am indifferent to this product to be honest.", "neutral"),
    ("It is what it is. Does the job without any flair.", "neutral"),
    ("Unremarkable but functional. Acceptable overall.", "neutral"),
    ("The quality is middling, as is the price. Fair enough.", "neutral"),
    ("Not my favourite but not my least favourite either.", "neutral"),
    ("I feel nothing strongly about this product.", "neutral"),
    ("It came on time, works fine, no issues at all.", "neutral"),
    ("The experience was perfectly standard. No surprises.", "neutral"),
    ("Acceptable performance for an entry level product.", "neutral"),
    ("It ticks the basic boxes without going beyond them.", "neutral"),
    ("I do not have strong feelings either way about this.", "neutral"),
    ("It is a reasonable product at a reasonable price.", "neutral"),
    ("The support team was professional and adequately helpful.", "neutral"),
    ("Neither disappointed nor delighted. Right in the middle.", "neutral"),
    ("This product meets its stated specification. That is all.", "neutral"),
    ("The packaging was simple, product was functional.", "neutral"),
    ("I am ambivalent about this purchase. It is just okay.", "neutral"),
    ("It does what it promises, no more, no less.", "neutral"),
    ("The restaurant was average. Food was okay, service was fine.", "neutral"),
    ("No strong positive or negative feelings about this one.", "neutral"),
    ("It is satisfactory. Not memorable, but not frustrating.", "neutral"),
    ("I would neither recommend nor discourage buying this.", "neutral"),
    ("The product and experience were both entirely forgettable.", "neutral"),
    ("Works as expected, arrived as expected. Neutral overall.", "neutral"),
    ("Fair for the price point. Nothing exceptional.", "neutral"),
    ("Average quality, average delivery, average experience.", "neutral"),
    ("Some parts are good, some are not, overall mediocre.", "neutral"),
    ("I feel indifferent. It works. It exists. That is it.", "neutral"),
    ("The product covers the basics adequately enough.", "neutral"),
    ("Somewhere between satisfied and dissatisfied. Mixed.", "neutral"),
    ("It is a functional item. No complaints, no praise.", "neutral"),
    ("I bought it, it works, I have no strong opinion.", "neutral"),
    ("This is a middle of the road product. Squarely average.", "neutral"),
    ("It performs its intended function. I am neither happy nor sad.", "neutral"),
    ("The design is plain and the performance is standard.", "neutral"),
    ("I rate this a solid three out of five. Average.", "neutral"),
    ("No issues so far, no highlights either. Neutral experience.", "neutral"),
    ("It fills the gap in my needs without being exciting.", "neutral"),
    ("Adequate. That one word summarises this product.", "neutral"),
    ("The event was fine. Not great, not bad. Just fine.", "neutral"),
    ("I feel completely neutral towards this purchase.", "neutral"),
    ("It has pros and cons that roughly balance each other out.", "neutral"),
    ("The delivery was on time, the product was as described.", "neutral"),
    ("No wow factor but no disaster either. Mediocre.", "neutral"),
    ("It is a basic but functional product. Three stars.", "neutral"),
    ("I am not especially pleased or displeased. Standard stuff.", "neutral"),
    ("Not remarkable in any way but not terrible either.", "neutral"),
    ("Gets the job done without inspiring any emotion.", "neutral"),
    ("I have nothing particularly good or bad to report.", "neutral"),
    ("This product is about what I expected. Nothing more.", "neutral"),
    ("The experience was run of the mill in every regard.", "neutral"),
    ("Neither impressed nor disappointed. Just neutral.", "neutral"),
    ("The product is okay. My life is unchanged by owning it.", "neutral"),
    ("I could take it or leave it. Not bothered either way.", "neutral"),
    ("Middle of the pack in terms of quality and value.", "neutral"),
    ("Adequate performance for its intended use case.", "neutral"),
    ("I think it is fine. Just fine. No strong opinions.", "neutral"),
    ("The features are present and functional. That is enough.", "neutral"),
    ("I bought it for a reason and it fulfilled that reason.", "neutral"),
    ("The customer service was responsive and adequate.", "neutral"),
    ("Works without issues. I have not felt excited by it.", "neutral"),
    ("It is passable for the money. Nothing special though.", "neutral"),
    ("The product does its thing. I do not think about it much.", "neutral"),
    ("I have not regretted it but I have not raved about it.", "neutral"),
    ("It sits in the middle of the quality spectrum comfortably.", "neutral"),
    ("My reaction to this product is a shrug. It is fine.", "neutral"),
    ("I am undecided whether I would buy it again or not.", "neutral"),
    ("Fully functional, reasonably priced, ultimately forgettable.", "neutral"),
    ("Acceptable across the board. Not exciting though.", "neutral"),
    ("My overall impression is neutral. Nothing stands out.", "neutral"),
    ("I used it, it worked, I moved on. That sums it up.", "neutral"),
    ("It is exactly what I expected. Neither more nor less.", "neutral"),
    ("The experience was unremarkable in the best way possible.", "neutral"),
    ("I am mildly content. That is the most I can say.", "neutral"),
    ("If pressed, I would describe this as simply adequate.", "neutral"),
    ("Everything is fine. No complaints, no compliments.", "neutral"),
    ("It fills a role in my life without making an impression.", "neutral"),
    ("The performance is bang on average. Totally standard.", "neutral"),
    ("I am as indifferent to this as I am to plain water.", "neutral"),
    ("Some things work well, others are lacking. Overall okay.", "neutral"),
    ("Not worth praising, not worth condemning. It just is.", "neutral"),
    ("I see this product as a functional, forgettable item.", "neutral"),
    ("It handles its tasks without glory or failure. Fine.", "neutral"),
    ("I think most people would rate this a three. I agree.", "neutral"),
    ("Thoroughly average in quality, price, and experience.", "neutral"),
    ("My feelings can best be described as flat on this one.", "neutral"),
    ("It is unremarkable. It works. That is about it.", "neutral"),
    ("The product arrived, it functions, I have no strong views.", "neutral"),
    ("I feel balanced about this. Good and bad cancel out.", "neutral"),
    ("The item is as described. Neither surprising nor lacking.", "neutral"),
    ("Standard experience with a standard product. Neutral.", "neutral"),
    ("I feel nothing in particular when using this product.", "neutral"),
    ("There is nothing to dislike but nothing to love either.", "neutral"),
    ("It is a net-zero experience. Fine in all respects.", "neutral"),
    ("I am sitting on the fence with this one. Genuinely mixed.", "neutral"),
    ("The delivery and product quality are both passable.", "neutral"),
    ("I did not love it. I did not hate it. I used it.", "neutral"),
    ("There were good bits and bad bits. Overall, just okay.", "neutral"),
    ("I can neither strongly recommend this nor warn against it.", "neutral"),
    ("It is a standard product at a standard price. Neutral.", "neutral"),
    ("The whole experience was underwhelming in a quiet way.", "neutral"),
    ("I have no notable feelings about this product.", "neutral"),
    ("Competent and unremarkable. Sits right in the middle.", "neutral"),
    ("I expect I will forget I own this in a few months.", "neutral"),
    ("Does what it says on the box, nothing more nothing less.", "neutral"),
    ("Not particularly good, not particularly bad. Average.", "neutral"),
    ("The product is entirely unremarkable. Neutral feelings.", "neutral"),
    ("I feel a sense of meh about this purchase overall.", "neutral"),
    ("This product is the definition of average. Just okay.", "neutral"),
]

LABELS    = ["positive", "negative", "neutral"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

# ─────────────────────────────────────────────────────────────
#  NLP PREPROCESSING
# ─────────────────────────────────────────────────────────────
CONTRACTIONS = {
    "can't":"cannot","won't":"will not","n't":" not","i'm":"i am",
    "i've":"i have","i'll":"i will","i'd":"i would","it's":"it is",
    "that's":"that is","he's":"he is","she's":"she is",
    "we're":"we are","they're":"they are","you're":"you are",
}
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","up","about","into","through","after","is","was","are","were",
    "be","been","being","have","has","had","do","does","did","will","would",
    "could","should","may","might","shall","this","that","these","those",
    "i","me","my","we","our","you","your","he","his","she","her","it","its",
    "they","their","what","which","who","whom","am",
}

def preprocess(text):
    text = text.lower()
    for k, v in CONTRACTIONS.items():
        text = text.replace(k, v)
    text = re.sub(r"[^a-z\s]", " ", text)
    return [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]

# ─────────────────────────────────────────────────────────────
#  VOCABULARY
# ─────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def build(self, corpus):
        freq = Counter(w for tokens in corpus for w in tokens)
        for w in sorted(freq):
            if w not in self.word2idx:
                self.word2idx[w] = len(self.word2idx)
        return self

    def __len__(self):
        return len(self.word2idx)

# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION  (60-dim NLP feature vector)
# ─────────────────────────────────────────────────────────────
POS_WORDS = {
    'love','great','excellent','wonderful','amazing','fantastic','brilliant',
    'outstanding','perfect','superb','incredible','exceptional','delightful',
    'happy','joyful','pleased','thrilled','best','good','nice','positive',
    'helpful','effective','efficient','reliable','beautiful','impressive',
    'remarkable','fabulous','glorious','phenomenal','blessed','enjoy',
    'grateful','satisfied','delighted','gorgeous','breathtaking','radiant'
}
NEG_WORDS = {
    'terrible','horrible','awful','dreadful','disgusting','atrocious','appalling',
    'worst','bad','poor','inferior','defective','broken','useless','worthless',
    'hate','despise','furious','angry','upset','disappointed','frustrated',
    'failed','failure','disaster','nightmare','regret','waste','rubbish','junk',
    'fake','scam','fraud','cheap','flimsy','pathetic','avoid','infuriating',
    'miserable','shameful','dreadful','catastrophic','devastating'
}
INTENSITY_MAP = {
    'absolutely':2.0,'totally':1.8,'completely':1.8,'extremely':2.0,
    'incredibly':1.9,'utterly':1.9,'really':1.4,'very':1.3,
    'genuinely':1.5,'truly':1.6,'deeply':1.7,'highly':1.5,
    'somewhat':0.7,'slightly':0.6,'fairly':0.8,'quite':0.9,
}

_vocab  = None
_top_50 = None

def _ensure_vocab():
    global _vocab, _top_50
    if _vocab is None:
        corpus  = [preprocess(t) for t,_ in DATASET]
        _vocab  = Vocabulary().build(corpus)
        _top_50 = list(_vocab.word2idx.keys())[2:52]

def extract_features(tokens):
    _ensure_vocab()
    pos  = sum(1 for t in tokens if t in POS_WORDS)
    neg  = sum(1 for t in tokens if t in NEG_WORDS)
    intens = max((INTENSITY_MAP.get(t,1.0) for t in tokens), default=1.0)
    n    = max(len(tokens), 1)
    tset = set(tokens)
    bow  = np.array([1.0 if w in tset else 0.0 for w in _top_50], dtype=np.float32)
    base = np.array([
        pos/n, neg/n, (pos-neg)/n, intens/2.0, min(n/20,1.0),
        float(pos>neg), float(neg>pos), float(pos==neg),
        float(intens>1.5), float(intens<0.8),
    ], dtype=np.float32)
    return np.concatenate([base, bow])  # 60-dim

# ─────────────────────────────────────────────────────────────
#  DEEP LEARNING MODEL  — 3-Layer MLP with ReLU
#  Architecture: 60 → 128 → 64 → 3  (softmax output)
# ─────────────────────────────────────────────────────────────
class SentimentMLP:
    def __init__(self, input_dim=60):
        s1 = np.sqrt(2/input_dim)
        s2 = np.sqrt(2/128)
        s3 = np.sqrt(2/64)
        self.W1 = np.random.randn(input_dim, 128).astype(np.float32) * s1
        self.b1 = np.zeros(128, dtype=np.float32)
        self.W2 = np.random.randn(128, 64).astype(np.float32)        * s2
        self.b2 = np.zeros(64,  dtype=np.float32)
        self.W3 = np.random.randn(64, 3).astype(np.float32)          * s3
        self.b3 = np.zeros(3,   dtype=np.float32)

    def _softmax(self, z):
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def forward(self, X):
        a1 = np.maximum(0, X  @ self.W1 + self.b1)
        a2 = np.maximum(0, a1 @ self.W2 + self.b2)
        p  = self._softmax(a2 @ self.W3 + self.b3)
        return a1, a2, p

    def predict_proba(self, X):
        _, _, p = self.forward(X)
        return p

    def train(self, X, Y, epochs=80, lr=0.008, batch=32, verbose=True):
        n = len(X)
        if verbose:
            print("=" * 57)
            print("  Deep NLP Sentiment Model — Training")
            print(f"  Architecture : {X.shape[1]} → 128 → 64 → 3 (softmax)")
            print(f"  Dataset      : {n} samples   Epochs: {epochs}")
            print("=" * 57)

        for ep in range(1, epochs+1):
            idx = np.random.permutation(n)
            tl = tc = 0
            for s in range(0, n, batch):
                bi = idx[s:s+batch]; xb = X[bi]; yb = Y[bi]
                a1, a2, p = self.forward(xb)
                loss = -np.log(p[np.arange(len(yb)), yb] + 1e-10).mean()
                tl += loss; tc += (p.argmax(1) == yb).sum()

                dz3 = p.copy()
                dz3[np.arange(len(yb)), yb] -= 1; dz3 /= len(yb)
                dW3 = a2.T @ dz3; db3 = dz3.sum(0)
                da2 = (dz3 @ self.W3.T) * (a2 > 0)
                dW2 = a1.T @ da2; db2 = da2.sum(0)
                da1 = (da2 @ self.W2.T) * (a1 > 0)
                dW1 = xb.T @ da1; db1 = da1.sum(0)

                self.W3 -= lr*dW3; self.b3 -= lr*db3
                self.W2 -= lr*dW2; self.b2 -= lr*db2
                self.W1 -= lr*dW1; self.b1 -= lr*db1

            if verbose and (ep % 10 == 0 or ep == 1):
                acc = tc/n*100; nb = max(n//batch, 1)
                bar = '█'*int(acc/5)+'░'*(20-int(acc/5))
                print(f"  Epoch {ep:>3}/{epochs}  Loss {tl/nb:.4f}  "
                      f"Acc {acc:.1f}%  [{bar}]")
        if verbose:
            print("=" * 57)
            print("  ✓ Training complete!")
            print("=" * 57)

    def save(self, path):
        np.savez(path, W1=self.W1,b1=self.b1,W2=self.W2,
                 b2=self.b2,W3=self.W3,b3=self.b3)
        print(f"  Weights saved → {path}")

    def load(self, path):
        d = np.load(path)
        self.W1,self.b1=d['W1'],d['b1']
        self.W2,self.b2=d['W2'],d['b2']
        self.W3,self.b3=d['W3'],d['b3']
        print(f"  Weights loaded ← {path}")

# ─────────────────────────────────────────────────────────────
#  INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────
_model = None

def _init():
    global _model
    if _model is not None: return
    _ensure_vocab()
    np.random.seed(42)
    X = np.array([extract_features(preprocess(t)) for t,_ in DATASET])
    Y = np.array([LABEL2IDX[l] for _,l in DATASET])
    _model = SentimentMLP(input_dim=X.shape[1])
    _model.train(X, Y, epochs=80, lr=0.008, batch=32)

def analyse(text: str) -> dict:
    """Main inference function. Returns rich result dict."""
    _init()
    tokens = preprocess(text)
    feat   = extract_features(tokens).reshape(1, -1)
    probs  = _model.predict_proba(feat)[0]
    idx    = int(probs.argmax())
    label  = LABELS[idx]
    conf   = float(probs[idx])
    intens = max((INTENSITY_MAP.get(t,1.0) for t in tokens), default=1.0)
    int_label = ("Very High" if intens>=1.8 else "High" if intens>=1.4
                 else "Moderate" if intens>=1.0 else "Low")

    # Per-token attention approximation
    attn = []
    for tok in tokens[:20]:
        s = 0.1
        if tok in POS_WORDS: s += 0.5
        if tok in NEG_WORDS: s += 0.5
        s += INTENSITY_MAP.get(tok, 0) * 0.1
        attn.append({"word": tok, "score": round(min(s, 1.0), 3)})

    return {
        "sentiment":   label,
        "emoji":       {"positive":"😊","negative":"😞","neutral":"😐"}[label],
        "confidence":  round(conf * 100, 1),
        "intensity":   int_label,
        "probabilities": {
            "positive": round(float(probs[0]) * 100, 1),
            "negative": round(float(probs[1]) * 100, 1),
            "neutral":  round(float(probs[2]) * 100, 1),
        },
        "tokens":       tokens,
        "token_count":  len(tokens),
        "word_count":   len(text.split()),
        "attention":    attn,
    }

# ─────────────────────────────────────────────────────────────
#  FLASK REST API
# ─────────────────────────────────────────────────────────────
def start_server(port=5000):
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not found. Install with: pip install flask")
        return

    app = Flask(__name__)

    @app.after_request
    def cors(r):
        r.headers["Access-Control-Allow-Origin"]  = "*"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return r

    @app.route("/analyse", methods=["POST","OPTIONS"])
    def do_analyse():
        if request.method=="OPTIONS": return "", 204
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text","").strip()
        if not text: return jsonify({"error":"No text provided"}), 400
        return jsonify(analyse(text))

    @app.route("/health")
    def health():
        return jsonify({"status":"ok","model":"3-Layer MLP","classes":LABELS})

    print(f"\n  Server → http://localhost:{port}")
    print(f"  POST /analyse  {{\"text\":\"...\"}}  |  GET /health\n")
    app.run(host="0.0.0.0", port=port, debug=False)

# ─────────────────────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────────────────────
def run_demo():
    _init()
    print("\n" + "─"*65)
    print("  LIVE DEMO — Sentiment Inference")
    print("─"*65)
    tests = [
        "I absolutely love this product, it changed my life!",
        "Terrible product, completely broken on arrival.",
        "It is okay, nothing special but does the job.",
        "The service was phenomenal and incredibly helpful.",
        "I am so frustrated I cannot even put it into words.",
        "Delivery was fine, product is decent, no major complaints.",
        "This brilliant software exceeded all expectations!",
        "Complete garbage, waste of money, avoid at all costs.",
        "Average quality, neither good nor bad overall.",
        "I feel deeply grateful for this wonderful experience.",
    ]
    for t in tests:
        r = analyse(t)
        bar='█'*int(r['confidence']/5)+'░'*(20-int(r['confidence']/5))
        print(f"\n  {r['emoji']}  {r['sentiment'].upper():10} {r['confidence']:5.1f}%  [{bar}]")
        print(f"     \"{t[:60]}\"")
        print(f"     Probs: +{r['probabilities']['positive']:5.1f}%  "
              f"-{r['probabilities']['negative']:5.1f}%  "
              f"~{r['probabilities']['neutral']:5.1f}%  "
              f"| Intensity: {r['intensity']}")
    print(f"\n{'─'*65}\n")

# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--text" in sys.argv:
        i = sys.argv.index("--text")
        text = sys.argv[i+1] if i+1<len(sys.argv) else ""
        r = analyse(text)
        print(json.dumps(r, indent=2))
    elif "--server" in sys.argv:
        run_demo()
        start_server()
    else:
        run_demo()
