Website Link: https://ec2-52-39-77-238.us-west-2.compute.amazonaws.com:8050

# ReviewAnalyzer
A Web App developed using Dash, Flask and Postgresql, it classifies user reviews into positive, negative and neutral. The user can type in his review of a particular brand and the model will output scores which indicates how positive the review is on the basis of a probability. This probability is then translated into a suggested rating on a scale from 1-5. Users can submit their reviews or try to review another brand.

### Web Scraping
The results were various product reviews scraped from trustpilot.com. Selenium was used to extract the URL's of brand review pages, and the Scrapy library was used to extract the reviews themselves. Around 28,000 reviews were extracted

### Deep Learning Model
The sentiment analyzer used is a Character-Based Convolutional Neural Network. The relevant reference for the model is: https://arxiv.org/pdf/1509.01626.pdf
The model was developed using TensorFlow.

