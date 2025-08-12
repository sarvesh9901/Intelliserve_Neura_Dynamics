# Chatbot Test Results (2025-08-12 12:06:47)

## Test 1: test_weather_in_mumbai
**Query:** tell me weather in mumbai

**Response:** Weather in Mumbai, IN: 27.88°C, feels like 31.09°C, scattered clouds.

**Status:** PASS

---

## Test 2: test_weather_in_invalid_city
**Query:** tell me weather in asdfghjkl

**Response:** Error: city not found

**Status:** PASS

---

## Test 3: test_find_match_known_query
**Query:** What is fact table?

**Response:** The fact table contains the transaction data (Sales Data). The same uniqueness rule that applies to dimension tables (at least one unique column) is not applicable to fact tables.

**Status:** PASS

---

## Test 4: test_find_match_unknown_query
**Query:** Tell me about aliens on Mars?

**Response:** I don't know.

**Status:** PASS

---

## Test 5: test_multiple_queries
**Query:** What is the weather in Pune?

**Response:** Weather in Pune, IN: 26.96°C, feels like 28.4°C, broken clouds.

**Status:** PASS

---

## Test 6: test_multiple_queries
**Query:** What is claim procedure for car insurance?

**Response:** I don't know.

**Status:** PASS

---

