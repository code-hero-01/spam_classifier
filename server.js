const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// ─── Training Data ────────────────────────────────────────────────────────────

const SPAM_EMAILS = [
  "Congratulations you have won a free prize click here to claim your reward now",
  "URGENT your account has been suspended verify your information immediately",
  "Make money fast work from home earn thousands per week guaranteed income",
  "Free Viagra discount pills no prescription needed order now special offer",
  "You have been selected for a special offer claim your free gift today",
  "Click here to win a brand new iPhone limited time offer act now",
  "Nigerian prince needs your help transfer millions get 30 percent commission",
  "Lose weight fast guaranteed results with our miracle diet pill buy now",
  "Your PayPal account is compromised log in immediately to secure your funds",
  "Hot singles in your area want to meet you click here to see profiles",
  "Earn money online fast easy guaranteed no experience needed start today",
  "FREE credit report check your score now special limited offer expires soon",
  "WINNER you have been selected claim your cash prize call this number now",
  "Buy cheap medications online no prescription discount pharmacy lowest prices",
  "Investment opportunity guaranteed returns triple your money in 30 days",
  "Urgent bank transfer needed help me move money reward you generously",
  "Exclusive deal for you today only massive discount shop now free shipping",
  "Casino bonus free spins win big jackpot play online poker slots now",
  "Debt relief program qualify today eliminate credit card debt immediately",
  "Work from home opportunity unlimited income potential no experience required",
  "You owe taxes IRS will arrest you call now to pay and avoid penalty",
  "Miracle cure doctors hate this secret weight loss trick shed pounds fast",
  "Claim your inheritance funds contact us verify identity bank details required",
  "Best mortgage rates refinance now save thousands call for free quote today",
  "Your subscription expires update billing information to continue service",
  "SALE 90 percent off luxury watches designer brands cheap authentic replica",
  "Your computer has virus download antivirus software protect yourself now",
  "Earn passive income cryptocurrency investment guaranteed profit daily",
  "Free samples of our new product just pay shipping order while supplies last",
  "Congratulations you qualified for pre-approved loan no credit check apply",
  "Hot deal exclusive offer for valued customer limited quantity grab yours now",
  "Verify your account immediately or it will be permanently closed urgent",
  "Join our network marketing team unlimited earning potential be your own boss",
  "Increase your size results guaranteed satisfaction or money back discreet",
  "Unclaimed package waiting for you pay small fee to release your shipment",
  "Secret investment club insider stock tips make thousands every week",
  "Your email address has won lottery prize contact agent to claim prize",
  "Cheap software license Windows Office antivirus at unbeatable low prices",
  "Adult dating site hot women waiting for you no strings attached tonight",
  "Payday loan instant approval no credit check get cash deposited today"
];

const HAM_EMAILS = [
  "Hi can we reschedule our meeting to Thursday afternoon works better for me",
  "Attached is the report you requested please review before the presentation",
  "Thanks for dinner last night it was great to catch up with everyone",
  "The project deadline has been moved to next Friday please adjust your schedule",
  "Just a reminder that the team lunch is tomorrow at noon in the break room",
  "I reviewed your pull request and left some comments please take a look",
  "Happy birthday hope you have a wonderful day with your family and friends",
  "Can you send me the quarterly report when you get a chance no rush",
  "The client approved the proposal we can start development next week",
  "Please find the invoice attached payment is due within 30 days thank you",
  "Our book club meets this Saturday at 3pm let me know if you can make it",
  "I will be out of office next week back on Monday if urgent contact Sarah",
  "The conference call is confirmed for 2pm dial in using the link below",
  "Just wanted to check in how are things going on your end let me know",
  "Your package has been shipped tracking number is attached estimated 3 days",
  "Great work on the presentation today the client was very impressed well done",
  "Can we grab coffee this week to discuss the new project I have some ideas",
  "The budget for next quarter has been approved attached please review details",
  "Kids school play is next Wednesday evening hope you can make it should be fun",
  "Just checking you received my last email about the contract please confirm",
  "The server maintenance is scheduled for Sunday night minimal downtime expected",
  "Congratulations on your promotion very well deserved you have worked so hard",
  "Here are the meeting notes from today please add any corrections or additions",
  "My flight lands at 6pm can you pick me up from the airport if convenient",
  "The recipe you asked for is below let me know how it turns out enjoy",
  "All team members please complete the annual review by end of this month",
  "We are having a small gathering Saturday evening you and family are welcome",
  "The library book you requested is now available please pick it up this week",
  "Just a heads up the gym will be closed Monday for holiday back Tuesday",
  "Loved the article you shared very insightful got me thinking about our strategy",
  "Can you review this draft before I send to the client any feedback welcome",
  "Reminder your dentist appointment is next Tuesday at 10am please confirm",
  "The new coffee machine is in the break room everyone feel free to use it",
  "I finished reading that book you recommended absolutely loved it thank you",
  "Please welcome our new team member joining us Monday from the Boston office",
  "The road will be closed Saturday for the marathon plan alternate routes",
  "Your feedback on the design was really helpful made the changes you suggested",
  "Mom called wants to know if you are coming for thanksgiving please let her know",
  "The software update is ready for testing please check it before release",
  "Thanks for covering my shift yesterday really appreciated I owe you one"
];

// ─── Naive Bayes Classifier ───────────────────────────────────────────────────

class NaiveBayesClassifier {
  constructor(alpha = 1) {
    this.alpha = alpha;           // Laplace smoothing parameter
    this.spamWordCounts = {};
    this.hamWordCounts = {};
    this.spamTokenTotal = 0;      // total tokens seen in spam
    this.hamTokenTotal = 0;       // total tokens seen in ham
    this.spamDocCount = 0;
    this.hamDocCount = 0;
    this.vocab = new Set();
  }

  // Convert raw text into lowercase word tokens
  tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z\s]/g, '')
      .split(/\s+/)
      .filter(w => w.length > 1);
  }

  // Train on an array of documents for a given label ('spam' | 'ham')
  train(docs, label) {
    docs.forEach(doc => {
      if (label === 'spam') this.spamDocCount++;
      else this.hamDocCount++;

      this.tokenize(doc).forEach(token => {
        this.vocab.add(token);
        if (label === 'spam') {
          this.spamWordCounts[token] = (this.spamWordCounts[token] || 0) + 1;
          this.spamTokenTotal++;
        } else {
          this.hamWordCounts[token] = (this.hamWordCounts[token] || 0) + 1;
          this.hamTokenTotal++;
        }
      });
    });
  }

  // Log P(word | label) with Laplace smoothing
  logWordLikelihood(word, label) {
    const counts = label === 'spam' ? this.spamWordCounts : this.hamWordCounts;
    const total  = label === 'spam' ? this.spamTokenTotal  : this.hamTokenTotal;
    return Math.log((counts[word] || 0) + this.alpha)
         - Math.log(total + this.alpha * this.vocab.size);
  }

  // Classify text; returns full breakdown for the API response
  classify(text) {
    const tokens = this.tokenize(text);
    const totalDocs = this.spamDocCount + this.hamDocCount;

    // Log priors
    const logPriorSpam = Math.log(this.spamDocCount / totalDocs);
    const logPriorHam  = Math.log(this.hamDocCount  / totalDocs);

    let logSpam = logPriorSpam;
    let logHam  = logPriorHam;

    const wordContributions = [];

    tokens.forEach(token => {
      if (this.vocab.has(token)) {
        const ls = this.logWordLikelihood(token, 'spam');
        const lh = this.logWordLikelihood(token, 'ham');
        logSpam += ls;
        logHam  += lh;
        wordContributions.push({ word: token, spamScore: ls, hamScore: lh, diff: ls - lh });
      }
    });

    // Convert log-probabilities to normalised probabilities (softmax)
    const maxLog  = Math.max(logSpam, logHam);
    const expSpam = Math.exp(logSpam - maxLog);
    const expHam  = Math.exp(logHam  - maxLog);
    const sum     = expSpam + expHam;
    const pSpam   = expSpam / sum;
    const pHam    = expHam  / sum;

    // Sort contributing words by absolute log-odds difference
    wordContributions.sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));

    return {
      label: pSpam >= 0.5 ? 'spam' : 'ham',
      pSpam: parseFloat(pSpam.toFixed(4)),
      pHam:  parseFloat(pHam.toFixed(4)),
      logSpam: parseFloat(logSpam.toFixed(4)),
      logHam:  parseFloat(logHam.toFixed(4)),
      logPriorSpam: parseFloat(logPriorSpam.toFixed(4)),
      logPriorHam:  parseFloat(logPriorHam.toFixed(4)),
      tokensFound: wordContributions.length,
      totalTokens: tokens.length,
      topWords: wordContributions.slice(0, 10),
    };
  }

  getStats() {
    return {
      vocabSize:     this.vocab.size,
      spamDocCount:  this.spamDocCount,
      hamDocCount:   this.hamDocCount,
      totalDocs:     this.spamDocCount + this.hamDocCount,
      spamTokenTotal: this.spamTokenTotal,
      hamTokenTotal:  this.hamTokenTotal,
      laplaceAlpha:  this.alpha,
    };
  }
}

// ─── Train once at startup ────────────────────────────────────────────────────

const classifier = new NaiveBayesClassifier(1);
classifier.train(SPAM_EMAILS, 'spam');
classifier.train(HAM_EMAILS,  'ham');
console.log('Model trained:', classifier.getStats());

// ─── Routes ───────────────────────────────────────────────────────────────────

// POST /classify  { text: "..." }
app.post('/classify', (req, res) => {
  const { text } = req.body;
  if (!text || typeof text !== 'string' || text.trim() === '') {
    return res.status(400).json({ error: 'Request body must include a non-empty "text" field.' });
  }
  const result = classifier.classify(text.trim());
  res.json(result);
});

// GET /stats
app.get('/stats', (req, res) => {
  res.json(classifier.getStats());
});

// Health check
// app.get('/', (req, res) => {
//   res.json({ status: 'ok', message: 'Naive Bayes Spam Classifier API is running.' });
// });

// ─── Start ────────────────────────────────────────────────────────────────────

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));