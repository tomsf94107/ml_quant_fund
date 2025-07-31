from forecast_utils import forecast_today_movement

msg, err = forecast_today_movement("MSFT")
print("✅ Forecast message:", msg)
print("❌ Error (if any):", err)
