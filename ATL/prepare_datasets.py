import sys
import os
sys.path.insert(0, '../../')

import ACDCDataManipulator as acdc

def generate_source(dataset_name, filename):
    if not os.path.isfile(filename + '_source.csv'):
        acdc.generate_csv_from_dataset(dataset_name, 5, True, False, 1)
        os.rename('source.csv', filename + '_source.csv')


def generate_target(dataset_name, filename):
    if not os.path.isfile(filename + '_target.csv'):
        acdc.generate_csv_from_dataset(dataset_name, 7, False, False, 1)
        os.rename('target.csv', filename + '_target.csv')


def generate_source_and_target(source_dataset, target_dataset, filename):
    generate_source(source_dataset, filename)
    generate_target(target_dataset, filename)

filename = 'usps_mnist'
generate_source_and_target('usps-16','mnist-16',filename)

filename = 'mnist_usps'
generate_source_and_target('mnist-28','usps-28',filename)

filename = 'amazon_review_beauty_luxury'
generate_source_and_target('amazon-review-all-beauty','amazon-review-luxury-beauty',filename)

filename = 'amazon_review_beauty_magazine'
generate_source_and_target('amazon-review-all-beauty','amazon-review-magazine-subscription',filename)

filename = 'amazon_review_beauty_books'
generate_source_and_target('amazon-review-all-beauty','amazon-review-books',filename)

filename = 'amazon_review_beauty_industrial'
generate_source_and_target('amazon-review-all-beauty','amazon-review-industrial-scientific',filename)

filename = 'amazon_review_luxury_beauty'
generate_source_and_target('amazon-review-luxury-beauty','amazon-review-all-beauty',filename)

filename = 'amazon_review_luxury_magazine'
generate_source_and_target('amazon-review-luxury-beauty','amazon-review-magazine-subscription',filename)

filename = 'amazon_review_luxury_books'
generate_source_and_target('amazon-review-luxury-beauty','amazon-review-books',filename)

filename = 'amazon_review_luxury_industrial'
generate_source_and_target('amazon-review-luxury-beauty','amazon-review-industrial-scientific',filename)

filename = 'amazon_review_books_beauty'
generate_source_and_target('amazon-review-books','amazon-review-all-beauty',filename)

filename = 'amazon_review_books_luxury'
generate_source_and_target('amazon-review-books','amazon-review-luxury-beauty',filename)

filename = 'amazon_review_books_magazine'
generate_source_and_target('amazon-review-books','amazon-review-magazine-subscription',filename)

filename = 'amazon_review_books_industrial'
generate_source_and_target('amazon-review-books','amazon-review-industrial-scientific',filename)

filename = 'amazon_review_industrial_beauty'
generate_source_and_target('amazon-review-industrial-scientific','amazon-review-all-beauty',filename)

filename = 'amazon_review_industrial_luxury'
generate_source_and_target('amazon-review-industrial-scientific','amazon-review-luxury-beauty',filename)

filename = 'amazon_review_industrial_magazine'
generate_source_and_target('amazon-review-industrial-scientific','amazon-review-magazine-subscription',filename)

filename = 'amazon_review_industrial_books'
generate_source_and_target('amazon-review-industrial-scientific','amazon-review-books',filename)

filename = 'amazon_review_magazine_beauty'
generate_source_and_target('amazon-review-magazine-subscription','amazon-review-all-beauty',filename)

filename = 'amazon_review_magazine_luxury'
generate_source_and_target('amazon-review-magazine-subscription','amazon-review-luxury-beauty',filename)

filename = 'amazon_review_magazine_industrial'
generate_source_and_target('amazon-review-magazine-subscription','amazon-review-industrial-scientific',filename)

filename = 'amazon_review_magazine_books'
generate_source_and_target('amazon-review-magazine-subscription','amazon-review-books',filename)