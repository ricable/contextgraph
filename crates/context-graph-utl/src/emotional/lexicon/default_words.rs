//! Default sentiment word lists for the lexicon.

use super::lexicon_impl::SentimentLexicon;

impl Default for SentimentLexicon {
    /// Create a default sentiment lexicon with common emotional words.
    ///
    /// Includes positive words like: excellent, wonderful, amazing, good, etc.
    /// Includes negative words like: terrible, awful, bad, disappointing, etc.
    fn default() -> Self {
        let mut lexicon = Self::new();

        // Highly positive words (0.8-1.0)
        for word in &[
            "excellent",
            "wonderful",
            "amazing",
            "fantastic",
            "brilliant",
            "outstanding",
            "perfect",
            "exceptional",
            "superb",
            "magnificent",
        ] {
            lexicon.add_positive(word, 0.9);
        }

        // Moderately positive words (0.5-0.7)
        for word in &[
            "good",
            "great",
            "nice",
            "pleasant",
            "lovely",
            "delightful",
            "happy",
            "glad",
            "pleased",
            "satisfied",
            "exciting",
            "interesting",
            "impressive",
            "remarkable",
            "valuable",
            "useful",
            "helpful",
        ] {
            lexicon.add_positive(word, 0.6);
        }

        // Mildly positive words (0.2-0.4)
        for word in &[
            "okay",
            "fine",
            "decent",
            "adequate",
            "acceptable",
            "reasonable",
            "positive",
            "favorable",
            "promising",
            "hopeful",
        ] {
            lexicon.add_positive(word, 0.3);
        }

        // Highly negative words (0.8-1.0)
        for word in &[
            "terrible",
            "awful",
            "horrible",
            "dreadful",
            "atrocious",
            "abysmal",
            "disastrous",
            "catastrophic",
            "devastating",
            "appalling",
        ] {
            lexicon.add_negative(word, 0.9);
        }

        // Moderately negative words (0.5-0.7)
        for word in &[
            "bad",
            "poor",
            "disappointing",
            "frustrating",
            "annoying",
            "unpleasant",
            "difficult",
            "problematic",
            "troublesome",
            "concerning",
            "worrying",
            "upsetting",
            "disturbing",
            "confusing",
            "unclear",
        ] {
            lexicon.add_negative(word, 0.6);
        }

        // Mildly negative words (0.2-0.4)
        for word in &[
            "mediocre",
            "subpar",
            "lacking",
            "insufficient",
            "underwhelming",
            "boring",
            "tedious",
            "dull",
            "unremarkable",
            "forgettable",
        ] {
            lexicon.add_negative(word, 0.3);
        }

        // Learning/discovery related positive words
        for word in &[
            "understand",
            "learned",
            "discovered",
            "realized",
            "grasped",
            "mastered",
            "comprehended",
            "insightful",
            "enlightening",
            "illuminating",
            "clarifying",
            "revealing",
            "informative",
            "educational",
        ] {
            lexicon.add_positive(word, 0.5);
        }

        // Cognitive struggle words (mildly negative)
        for word in &[
            "confused",
            "puzzled",
            "bewildered",
            "perplexed",
            "uncertain",
            "unsure",
            "doubtful",
            "struggling",
            "lost",
        ] {
            lexicon.add_negative(word, 0.4);
        }

        lexicon
    }
}
