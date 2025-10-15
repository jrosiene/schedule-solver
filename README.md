# Swap Planner

A web application for finding and planning shift swaps between healthcare workers.

## Features

- 📅 **Calendar View**: Visual calendar showing your shifts and available swap options
- 🔄 **2-Way Swaps**: Find direct shift swaps with other staff members
- 🔗 **3-Way Swaps**: Plan complex multi-person shift chains
- 🎯 **Smart Filtering**: Filter by date range, shift types, and specific people
- 📊 **Visual Planning**: Color-coded calendar with swap status indicators
- 📤 **Export**: Download your swap plans as CSV files

## How to Use

1. **Upload Schedule**: Upload your `.ics` schedule file or use the default one
2. **Select Yourself**: Choose your name from the dropdown
3. **Choose Partners**: Select other staff members you want to swap with
4. **Plan Swaps**: 
   - Click on your shifts (blue buttons) to see swap options
   - Click on green swap options to add them to your plan
   - Use "Lock In" to confirm swaps and enable 3-way chains
5. **Export Plan**: Download your final swap plan as a CSV

## Calendar Colors

- 🔵 **Blue buttons** = Your shifts with swap options available
- ⚪ **Gray buttons** = Your shifts with no swap options  
- 🟢 **Green buttons** = Available swap dates (shown when you select a shift)
- 🔷 **Dark blue buttons** = Shifts you're receiving (locked in)
- 🟢 **Dark green buttons** = Shifts you're giving away (locked in)

## Technical Details

- Built with Python Streamlit
- Parses ICS calendar files
- Supports complex shift compatibility rules
- Optimized for healthcare scheduling workflows

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Deployment

This application is designed to be deployed on Streamlit Cloud for easy sharing with colleagues.