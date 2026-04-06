#!/bin/bash
sqlite3 ~/Desktop/ML_Quant_Fund/data/sentiment.db "SELECT score_date, COUNT(*), ROUND(AVG(sentiment_score),2), SUM(CASE WHEN sentiment_label='BULLISH' THEN 1 ELSE 0 END), SUM(CASE WHEN sentiment_label='BEARISH' THEN 1 ELSE 0 END) FROM monday_sentiment WHERE score_date >= date('now','-7 days') GROUP BY score_date ORDER BY score_date DESC;"
