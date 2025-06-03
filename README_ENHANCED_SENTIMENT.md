# Sentiment Analysis Enhancement Project

## Overview

This project enhances the basic NLTK-based sentiment analysis script by integrating real-world datasets to provide a more robust and balanced set of example sentences. The enhanced version overcomes limitations in the original script by incorporating diverse datasets and implementing intelligent sentence classification.

## Comparison: Original vs. Enhanced Version

### Original Script (`sentiment_analysis_with_nltk.py`)
- Uses NLTK's subjectivity corpus for training and testing
- Includes a limited set of hard-coded example sentences (~40 total)
- Relies on manually curated example sentences with limited diversity
- No fallback mechanisms if resources are unavailable
- Limited scalability for testing with larger sentence sets

### Enhanced Script (`sentiment_analysis_enhanced.py`)
- Maintains the original NLTK subjectivity corpus for training
- Dynamically loads thousands of sentences from public sentiment datasets
- Implements smart classification to separate "regular" vs "tricky" sentences
- Features a robust caching system for dataset storage
- Includes fallback mechanisms when network resources are unavailable
- Scales to handle 1000+ sentences while maintaining performance

## Key Enhancements

1. **Dynamic Dataset Integration**
   - Connects to multiple stable, open datasets (IMDB, Amazon Reviews, Twitter, etc.)
   - Implements a local caching system to prevent repeated downloads
   - Uses multithreaded downloading to improve performance

2. **Intelligent Sentence Classification**
   - Categorizes sentences as "regular" (straightforward sentiment) or "tricky" (ambiguous)
   - Uses linguistic markers (negations, contrasts, sentence length) to identify challenging cases
   - Maintains balance between positive/negative examples

3. **Robustness Improvements**
   - Fallback sentence sets if network resources are unavailable
   - Graceful degradation when datasets can't be accessed
   - Dataset validation and sanitization

4. **Performance Optimizations**
   - Implements limits on total sentences to prevent memory/CPU overload
   - Maintains proportional representation when downsampling
   - Caches processed datasets for faster loading

## Dataset Sources

The enhanced script draws from multiple public sentiment datasets:

- **IMDB Movie Reviews** - Large movie review dataset with binary sentiment labels
- **Amazon Reviews** - Consumer product reviews with ratings
- **Stanford Movie Reviews** - Curated movie reviews dataset
- **Twitter Sentiment** - Social media sentiment samples
- **RT Movie Reviews** - Rotten Tomatoes movie reviews (positive and negative)

## Sentence Classification Criteria

### Regular Sentences
- Short to medium length (typically < 100 characters)
- Clear positive or negative sentiment
- Simple grammatical structure
- Equal balance of positive and negative examples

### Tricky Sentences
- Contains contrast words ("but", "however", "although", etc.)
- Contains negations ("not", "never", "no", etc.)
- Mixed sentiment words (both positive and negative terms)
- Complex grammatical structures
- Longer sentences (typically > 20 words)
- Sarcasm and irony indicators

## Implementation Details

### Caching System
The enhanced script implements a sophisticated caching system:
- Creates a `datasets_cache` directory to store downloaded datasets
- Uses compressed pickle files (.pkl.gz) to minimize storage requirements
- Checks cache validity before attempting new downloads

### Text Cleaning
Comprehensive text cleaning pipeline:
- Removes URLs, HTML tags, and special characters
- Normalizes contractions and punctuation
- Filters out extremely short or uninformative sentences

### Balancing Strategy
- Maintains equal numbers of positive and negative regular sentences
- Categorizes tricky sentences into subgroups (contrasts, negations, etc.)
- Samples proportionally from each subgroup to ensure diversity

## Usage Instructions

Run the enhanced script with the following options:

```bash
python sentiment_analysis_enhanced.py [--n_instances NUM] [--version VERSION_ID]
```

Where:
- `NUM` is the number of instances to load from the subjectivity corpus (default: 1000)
- `VERSION_ID` is an identifier for the output file (default: "1")

## Performance Considerations

- The script dynamically adjusts the number of sentences processed based on available resources
- Default configuration attempts to load 1000 regular and 1000 tricky sentences
- If resources are constrained, it will automatically downsample to a manageable number (default max: 2000 total)
- Processing time scales with the number of sentences analyzed

## Conclusion

The enhanced sentiment analysis script provides a more robust, diverse, and representative set of example sentences for sentiment analysis tasks. By incorporating real-world datasets and intelligent classification, it offers significant advantages over the original implementation while maintaining compatibility with the core NLTK-based analysis workflow.
