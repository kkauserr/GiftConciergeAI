INPUT_FILE = "/scratch/jkauser/gift-concierge/data/processed_products.jsonl"
OUTPUT_FILE = "DATA/samples/sample_corpus_v2.jsonl"

# Final v2 retained counts
BOOKS_COUNT = 3_266_409
HANDMADE_COUNT = 94_173

# Take 300 products from the beginning of each domain.
books_start = 1
handmade_start = BOOKS_COUNT + 1
toys_start = BOOKS_COUNT + HANDMADE_COUNT + 1

wanted_ranges = [
    (books_start, books_start + 299),
    (handmade_start, handmade_start + 299),
    (toys_start, toys_start + 299),
]

written = 0
range_index = 0

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line_number, line in enumerate(infile, start=1):

        if range_index >= len(wanted_ranges):
            break

        start, end = wanted_ranges[range_index]

        if start <= line_number <= end:
            outfile.write(line)
            written += 1

        if line_number >= end:
            range_index += 1

print("===== SAMPLE CORPUS CREATED =====")
print("Books:              300")
print("Handmade Products:  300")
print("Toys & Games:       300")
print(f"Total products:     {written}")
print(f"Output file:        {OUTPUT_FILE}")
