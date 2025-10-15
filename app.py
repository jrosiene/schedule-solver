"""
Shift Swap Finder - Python Streamlit Application
Finds possible 2-way and 3-way shift swaps in workplace schedules
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date
import datetime as dt
import io
from icalendar import Calendar
import itertools
from typing import List, Dict, Tuple, Optional
import re

# Page configuration
st.set_page_config(
    page_title="Shift Swap Finder",
    page_icon="🔄",
    layout="wide"
)

def parse_summary(summary: str) -> Tuple[str, str]:
    """
    Parse ICS summary to extract shift type and person name.
    Handles various formats like:
    - "Med A: Sutton, Paul"
    - "AM Swing/Med G: Rosse, Chris" 
    - "1/2 Clinic: Chi, Frank"
    - "RISK 1: Deshpande, Neha"
    - "CMP" (no person)
    """
    if not summary or summary.strip() == "":
        return "Unknown", "Unknown"
    
    # Remove any leading/trailing whitespace
    summary = summary.strip()
    
    # Handle the colon separator
    if ': ' not in summary:
        # If no colon, treat entire string as shift type (no person)
        return summary, "No Person"
    
    # Split on the first colon
    parts = summary.split(': ', 1)
    if len(parts) != 2:
        return summary, "No Person"
    
    shift_type = parts[0].strip()
    person_part = parts[1].strip()
    
    # Skip if person part is empty or just whitespace
    if not person_part or person_part.strip() == "":
        return shift_type, "No Person"
    
    # Clean up person name - handle escaped commas and other formatting
    # Remove backslashes before commas (from ICS escaping)
    person_part = person_part.replace('\\,', ',')
    
    # Handle cases where there might be multiple colons in the person name
    # Look for patterns like "LastName, FirstName" or "LastName FirstName"
    person = clean_person_name(person_part)
    
    return shift_type, person

def clean_person_name(person_part: str) -> str:
    """
    Clean and standardize person names with comprehensive handling.
    """
    if not person_part:
        return "Unknown"
    
    # Remove any extra whitespace
    person_part = person_part.strip()
    
    # Remove "(See times)" and similar annotations
    person_part = re.sub(r'\s*\([^)]*\)\s*$', '', person_part)
    
    # Custom name equivalencies
    name_mappings = {
        "Abay": "Rebecca Abay",
        "Bajwa": "Poornima Bajwa", 
        "Benson": "Doug Benson",
        "Chen": "Tiffany Chen",
        "ChenA": "Tiffany Chen",
        "Chi": "Frank Chi",
        "Haselden": "Lindsay Haselden",
        "Hoy": "Alex Hoy",
        "Kaplan": "Elizabeth Kaplan",
        "LaMotte": "Eric Lamotte",
        "Lande": "Rachel Lande",
        "Levin": "Jesse Levin",
        "Mayo": "Mark Mayo",
        "Mookherjee": "Som Mookherjee",
        "Mullins": "Elizabeth Mullins",
        "Pittenger": "Brook Pittenger",
        "Segerson": "K Segerson",
        "Shen": "Carlita Shen",
        "Shepherd": "Amanda Shepherd",
        "Trudeau": "Brittany Trudeau",
        "Young": "Scott Young",
        "Zafar": "Zaeema Zafar"
    }
    
    # Check for exact matches first
    if person_part in name_mappings:
        return name_mappings[person_part]
    
    # Handle multiple people (comma-separated) - take the first one
    if ',' in person_part:
        # Check if this looks like multiple people (has multiple commas)
        parts = person_part.split(',')
        if len(parts) > 2:  # Multiple people, take first
            first_person = parts[0].strip()
            # Check if first person matches any of our mappings
            if first_person in name_mappings:
                return name_mappings[first_person]
            # Handle "LastName, FirstName" format for first person
            if ',' in first_person:
                name_parts = first_person.split(',', 1)
                if len(name_parts) == 2:
                    last_name = name_parts[0].strip()
                    first_name = name_parts[1].strip()
                    return f"{first_name} {last_name}"
            return first_person
        else:
            # Single person with "LastName, FirstName" format
            parts = person_part.split(',', 1)
            if len(parts) == 2:
                last_name = parts[0].strip()
                first_name = parts[1].strip()
                # Check if the last name matches any of our mappings
                if last_name in name_mappings:
                    return name_mappings[last_name]
                return f"{first_name} {last_name}"
    
    # If no comma, return as-is
    return person_part

def update_shift_ownership(locked_swaps, original_shifts_by_date):
    """Update shift ownership based on locked swaps"""
    ownership = {}
    
    # Initialize with original ownership
    for date_obj, shift in original_shifts_by_date.items():
        ownership[date_obj] = {
            'person': shift.get('person', 'Unknown'),
            'shift_type': shift.get('shift_type', 'Unknown'),
            'start_time': shift.get('start_time', ''),
            'end_time': shift.get('end_time', '')
        }
    
    # Apply swaps to update ownership
    for swap in locked_swaps:
        your_date = swap['your_date']
        their_date = swap['their_date']
        
        # You give away your shift, receive their shift
        if their_date in ownership:
            # You now own their shift
            ownership[their_date] = {
                'person': swap['target_person'],  # You now own this shift
                'shift_type': swap['their_shift'].split(' ', 1)[1],  # Extract shift type
                'start_time': ownership[their_date]['start_time'],
                'end_time': ownership[their_date]['end_time']
            }
        
        # They now own your shift (for display purposes)
        if your_date in ownership:
            ownership[your_date] = {
                'person': swap['target_person'],  # They now own your shift
                'shift_type': ownership[your_date]['shift_type'],
                'start_time': ownership[your_date]['start_time'],
                'end_time': ownership[your_date]['end_time']
            }
    
    return ownership


def parse_ics_file(uploaded_file) -> pd.DataFrame:
    """Parse ICS file and extract shift data"""
    try:
        # Read the uploaded file
        file_content = uploaded_file.read()
        
        # Parse with icalendar
        cal = Calendar.from_ical(file_content)
        
        shifts = []
        
        for component in cal.walk():
            if component.name == "VEVENT":
                # Extract event data
                summary = str(component.get('summary', ''))
                start_dt = component.get('dtstart')
                end_dt = component.get('dtend')
                
                if start_dt and end_dt and summary:
                    # Parse summary with more sophisticated logic
                    shift_type, person = parse_summary(summary)
                    
                    # Convert dates
                    start_date = start_dt.dt.date() if hasattr(start_dt.dt, 'date') else start_dt.dt
                    start_time = start_dt.dt.time() if hasattr(start_dt.dt, 'time') else start_dt.dt
                    end_time = end_dt.dt.time() if hasattr(end_dt.dt, 'time') else end_dt.dt
                    
                    # Only include shifts with actual people
                    if person != "No Person" and person != "Unknown":
                        shifts.append({
                            'date': start_date,
                            'shift_type': shift_type,
                            'person': person,
                            'start_time': start_time.strftime('%H:%M') if start_time else '',
                            'end_time': end_time.strftime('%H:%M') if end_time else ''
                        })
        
        # Show parsing summary
        if shifts:
            st.info(f"✅ Successfully parsed {len(shifts)} shifts")
        else:
            st.warning("⚠️ No shifts found in the file")
        
        return pd.DataFrame(shifts)
    
    except Exception as e:
        st.error(f"Error parsing ICS file: {str(e)}")
        return pd.DataFrame()

def find_two_way_swaps(df: pd.DataFrame, shift_compatibility: Dict[str, List[str]]) -> pd.DataFrame:
    """Find all possible 2-way swaps"""
    swaps = []
    
    # Get all people
    people = df['person'].unique()
    
    # Check all pairs of people
    for person1, person2 in itertools.combinations(people, 2):
        person1_shifts = df[df['person'] == person1]
        person2_shifts = df[df['person'] == person2]
        
        # Check all combinations of their shifts
        for _, shift1 in person1_shifts.iterrows():
            for _, shift2 in person2_shifts.iterrows():
                # Check if swap is valid
                if is_valid_swap(shift1, shift2, shift_compatibility):
                    swaps.append({
                        'person1': person1,
                        'date1': shift1['date'],
                        'shift1': shift1['shift_type'],
                        'person2': person2,
                        'date2': shift2['date'],
                        'shift2': shift2['shift_type']
                    })
    
    return pd.DataFrame(swaps)

def find_three_way_swaps(df: pd.DataFrame, shift_compatibility: Dict[str, List[str]]) -> pd.DataFrame:
    """Find all possible 3-way swaps"""
    swaps = []
    
    # Get all people
    people = df['person'].unique()
    
    # Check all triplets of people
    for person1, person2, person3 in itertools.combinations(people, 3):
        person1_shifts = df[df['person'] == person1]
        person2_shifts = df[df['person'] == person2]
        person3_shifts = df[df['person'] == person3]
        
        # Check all combinations
        for _, shift1 in person1_shifts.iterrows():
            for _, shift2 in person2_shifts.iterrows():
                for _, shift3 in person3_shifts.iterrows():
                    # Check if 3-way swap is valid
                    if is_valid_three_way_swap(shift1, shift2, shift3, shift_compatibility):
                        swaps.append({
                            'person1': person1,
                            'date1': shift1['date'],
                            'shift1': shift1['shift_type'],
                            'person2': person2,
                            'date2': shift2['date'],
                            'shift2': shift2['shift_type'],
                            'person3': person3,
                            'date3': shift3['date'],
                            'shift3': shift3['shift_type']
                        })
    
    return pd.DataFrame(swaps)

def is_valid_swap(shift1: pd.Series, shift2: pd.Series, shift_compatibility: Dict[str, List[str]]) -> bool:
    """Check if a 2-way swap is valid"""
    # Can't swap with yourself
    if shift1['person'] == shift2['person']:
        return False
    
    # Check shift type compatibility
    shift1_type = shift1['shift_type']
    shift2_type = shift2['shift_type']
    
    # Check if shift1 can be taken by person2
    if not is_shift_compatible(shift1_type, shift2_type, shift_compatibility):
        return False
    
    # Check if shift2 can be taken by person1
    if not is_shift_compatible(shift2_type, shift1_type, shift_compatibility):
        return False
    
    return True

def is_valid_three_way_swap(shift1: pd.Series, shift2: pd.Series, shift3: pd.Series, 
                           shift_compatibility: Dict[str, List[str]]) -> bool:
    """Check if a 3-way swap is valid"""
    # All people must be different
    people = [shift1['person'], shift2['person'], shift3['person']]
    if len(set(people)) != 3:
        return False
    
    # Check circular compatibility: person2 can take shift1, person3 can take shift2, person1 can take shift3
    if not is_shift_compatible(shift1['shift_type'], shift2['shift_type'], shift_compatibility):
        return False
    if not is_shift_compatible(shift2['shift_type'], shift3['shift_type'], shift_compatibility):
        return False
    if not is_shift_compatible(shift3['shift_type'], shift1['shift_type'], shift_compatibility):
        return False
    
    return True

def is_shift_compatible(from_shift: str, to_shift: str, shift_compatibility: Dict[str, List[str]]) -> bool:
    """Check if two shift types are compatible"""
    # If no compatibility rules defined, allow all swaps
    if not shift_compatibility:
        return True
    
    # Check if to_shift is in the compatible list for from_shift
    if from_shift in shift_compatibility:
        return to_shift in shift_compatibility[from_shift]
    
    # If from_shift not in compatibility rules, assume compatible
    return True

def create_calendar_view(df: pd.DataFrame, primary_user: str, target_users: list, shift_compatibility: dict, locked_swaps: list = None) -> pd.DataFrame:
    """Create a calendar view showing potential swaps - ULTRA OPTIMIZED VERSION"""
    try:
        # Get primary user's shifts
        primary_shifts = df[df['person'] == primary_user].copy()
        
        # Add shifts that primary user now owns after locked swaps
        if locked_swaps:
            for swap in locked_swaps:
                their_date = swap['their_date']
                # Find the shift they're receiving
                received_shift = df[df['date'] == their_date].copy()
                if not received_shift.empty:
                    # Update the person to be the primary user
                    received_shift = received_shift.iloc[0].copy()
                    received_shift['person'] = primary_user
                    received_shift['shift_type'] = swap['their_shift'].split(' ', 1)[1]
                    primary_shifts = pd.concat([primary_shifts, received_shift.to_frame().T], ignore_index=True)
        
        if primary_shifts.empty:
            return pd.DataFrame()
        
        # Get target users' shifts (exclude primary user)
        target_shifts = df[df['person'].isin(target_users) & (df['person'] != primary_user)].copy()
        
        if target_shifts.empty:
            return pd.DataFrame()
        
        # ULTRA OPTIMIZATION: Pre-compute all data structures
        primary_scheduled_dates = set(primary_shifts['date'].tolist())
        
        # Create a lookup for compatible shifts by primary shift type
        compatible_shifts_lookup = {}
        for primary_shift_type in primary_shifts['shift_type'].unique():
            if primary_shift_type in shift_compatibility:
                compatible_shifts_lookup[primary_shift_type] = set(shift_compatibility[primary_shift_type])
            else:
                compatible_shifts_lookup[primary_shift_type] = set()
        
        # Create calendar data using vectorized operations
        calendar_data = []
        
        for _, primary_shift in primary_shifts.iterrows():
            compatible_shifts = []
            primary_shift_type = primary_shift['shift_type']
            primary_date = primary_shift['date']
            
            # Get compatible shift types for this primary shift
            compatible_types = compatible_shifts_lookup.get(primary_shift_type, set())
            
            if compatible_types:
                # Filter target shifts to only compatible types and dates where primary is NOT scheduled
                valid_targets = target_shifts[
                    (target_shifts['shift_type'].isin(compatible_types)) &
                    (~target_shifts['date'].isin(primary_scheduled_dates))
                ]
                
                # Convert to list of dictionaries efficiently
                for _, target_shift in valid_targets.iterrows():
                    compatible_shifts.append({
                        'target_person': target_shift['person'],
                        'target_shift_type': target_shift['shift_type'],
                        'target_date': target_shift['date']
                    })
            
            calendar_data.append({
                'date': primary_shift['date'],
                'shift_type': primary_shift['shift_type'],
                'person': primary_shift['person'],
                'start_time': primary_shift['start_time'],
                'end_time': primary_shift['end_time'],
                'compatible_swaps': compatible_shifts
            })
        
        return pd.DataFrame(calendar_data)
    
    except Exception as e:
        st.error(f"Error creating calendar view: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("Swap Planner")
    
    # Initialize session state
    if 'selected_date' not in st.session_state:
        st.session_state.selected_date = None
    if 'locked_swaps' not in st.session_state:
        st.session_state.locked_swaps = []
    if 'current_swap_plan' not in st.session_state:
        st.session_state.current_swap_plan = []
    if 'shift_ownership' not in st.session_state:
        st.session_state.shift_ownership = {}  # Track who owns each shift after swaps
    if 'cached_df' not in st.session_state:
        st.session_state.cached_df = None
    if 'cached_file_hash' not in st.session_state:
        st.session_state.cached_file_hash = None
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📋 Setup")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload ICS Schedule File",
            type=['ics'],
            help="Upload your .ics schedule file"
        )
        
        # Use default file if no upload
        if uploaded_file is None:
            try:
                with open("exportall.ics", "rb") as f:
                    uploaded_file = io.BytesIO(f.read())
                    uploaded_file.name = "exportall.ics"
            except FileNotFoundError:
                st.warning("No default file found. Please upload an ICS file.")
                return
        
        # Parse the file with caching
        if uploaded_file:
            # Create a hash of the file content for caching
            file_content = uploaded_file.read()
            file_hash = hash(file_content)
            
            # Check if we have cached data for this file
            if (st.session_state.cached_df is not None and 
                st.session_state.cached_file_hash == file_hash):
                df = st.session_state.cached_df
                st.success(f"✅ Using cached data - {len(df)} shifts")
            else:
                # Parse the file
                uploaded_file.seek(0)  # Reset file pointer
                with st.spinner("Parsing ICS file..."):
                    df = parse_ics_file(uploaded_file)
                
                if df.empty:
                    st.error("No data found in the ICS file")
                    return
                
                # Cache the results
                st.session_state.cached_df = df
                st.session_state.cached_file_hash = file_hash
                st.success(f"✅ Loaded {len(df)} shifts")
            
            # Primary user selection
            st.subheader("👤 Your Shifts")
            all_people = sorted(df['person'].unique())
            primary_user = st.selectbox(
                "Select yourself",
                all_people,
                help="Choose your name to see your shifts and potential swaps"
            )
            
            # Get primary user's shifts
            primary_shifts = df[df['person'] == primary_user]
            
            if primary_shifts.empty:
                st.warning(f"No shifts found for {primary_user}")
                return
            
            # Date range filter
            st.subheader("📅 Date Range Filter")
            min_date = df['date'].min()
            max_date = df['date'].max()
            
            # Default start date to today, but don't go before min_date
            default_start = max(dt.date.today(), min_date)
            date_range = st.date_input(
                "Select date range",
                value=(default_start, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Filter shifts to this date range"
            )
            
            # Filter by date range
            if len(date_range) == 2:
                df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
                st.info(f"Filtered to {len(df)} shifts between {date_range[0]} and {date_range[1]}")
            
            # Target users selection
            st.subheader("👥 Swap Partners")
            other_people = [p for p in all_people if p != primary_user]
            target_users = st.multiselect(
                "Select people you want to swap with",
                other_people,
                help="Choose people you're willing to swap shifts with"
            )
            
            # Shift compatibility settings
            st.subheader("⚙️ Swap Rules")
            allow_cross_type = st.checkbox(
                "Allow cross-category swaps",
                value=False,
                help="Allow swaps between Night, Day, and RISK shifts"
            )
            
            # Build compatibility matrix
            shift_types = sorted(df['shift_type'].unique())
            night_shifts = [s for s in shift_types if any(keyword in s.upper() for keyword in ['NOC', 'NIGHT', 'NIGHTS'])]
            risk_shifts = [s for s in shift_types if 'RISK' in s.upper()]
            day_shifts = [s for s in shift_types if s not in night_shifts and s not in risk_shifts]
            
            shift_compatibility = {}
            for shift_type in shift_types:
                compatible_shifts = []
                
                # Determine which category this shift belongs to
                if shift_type in night_shifts:
                    same_category = night_shifts
                elif shift_type in risk_shifts:
                    same_category = risk_shifts
                else:
                    same_category = day_shifts
                
                # Always allow same-category swaps
                compatible_shifts.extend([s for s in same_category if s != shift_type])
                
                # Add cross-type swaps if allowed
                if allow_cross_type:
                    if shift_type in night_shifts:
                        compatible_shifts.extend(risk_shifts + day_shifts)
                    elif shift_type in risk_shifts:
                        compatible_shifts.extend(night_shifts + day_shifts)
                    else:  # day shift
                        compatible_shifts.extend(night_shifts + risk_shifts)
                
                shift_compatibility[shift_type] = compatible_shifts
    
    # Main content area
    if uploaded_file and not df.empty:
        # Check if we have the required selections
        if 'primary_user' in locals() and 'target_users' in locals() and primary_user and target_users:
            st.header(f"📅 Swap Calendar for {primary_user}")
            
            # Create calendar view with progress indicator
            with st.spinner("Finding potential swaps..."):
                calendar_df = create_calendar_view(df, primary_user, target_users, shift_compatibility, st.session_state.locked_swaps)
        
            if not calendar_df.empty:
                # Create a proper calendar grid
                import calendar
                from datetime import datetime, timedelta
                
                # Get date range for calendar
                min_date = calendar_df['date'].min()
                max_date = calendar_df['date'].max()
                
                # Create a mapping of dates to shifts (allow multiple shifts per date)
                shifts_by_date = {}
                for _, shift in calendar_df.iterrows():
                    date_obj = shift['date']
                    if date_obj not in shifts_by_date:
                        shifts_by_date[date_obj] = []
                    shifts_by_date[date_obj].append(shift)
                
                # Update shift ownership based on locked swaps
                if st.session_state.locked_swaps:
                    updated_ownership = update_shift_ownership(st.session_state.locked_swaps, shifts_by_date)
                    # Update the shifts_by_date with new ownership
                    for date_obj, ownership_info in updated_ownership.items():
                        if date_obj in shifts_by_date:
                            for shift in shifts_by_date[date_obj]:
                                shift['person'] = ownership_info['person']
                                shift['shift_type'] = ownership_info['shift_type']
                
                
                # Calculate all months that contain shifts
                from datetime import date
                all_dates = [shift['date'] for _, shift in calendar_df.iterrows()]
                
                if all_dates:
                    min_shift_date = min(all_dates)
                    max_shift_date = max(all_dates)
                    
                    # Start from current date or earliest shift date, whichever is later
                    start_date = max(date.today(), min_shift_date)
                    
                    # Calculate months to show - ensure we cover all months with shifts
                    current_date = start_date
                    # Calculate the difference in months more accurately
                    year_diff = max_shift_date.year - start_date.year
                    month_diff = max_shift_date.month - start_date.month
                    months_to_show = max(2, year_diff * 12 + month_diff + 1)
                    
                else:
                    current_date = date.today()
                    months_to_show = 2
                
                # Helper function to check if a date is affected by locked swaps
                def is_date_affected_by_swap(date_obj, locked_swaps):
                    """Check if a date is affected by any locked swap"""
                    for swap in locked_swaps:
                        if date_obj == swap['your_date'] or date_obj == swap['their_date']:
                            return True, swap
                    return False, None
                
                # Create calendar grid with scrollable container
                
                # Add custom CSS for button colors
                st.markdown("""
                <style>
                /* Override Streamlit button colors */
                .stButton > button {
                    background-color: #1f77b4 !important;
                    color: white !important;
                    border: none !important;
                }
                .stButton > button:hover {
                    background-color: #0d47a1 !important;
                }
                /* Custom classes for different button types */
                .blue-button {
                    background-color: #1f77b4 !important;
                    color: white !important;
                }
                .gray-button {
                    background-color: #d3d3d3 !important;
                    color: black !important;
                }
                .green-button {
                    background-color: #2ca02c !important;
                    color: white !important;
                }
                .dark-blue-button {
                    background-color: #0d47a1 !important;
                    color: white !important;
                }
                .dark-green-button {
                    background-color: #1b5e20 !important;
                    color: white !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Add color legend
                st.info("""
                **Calendar Color Guide:**
                - 🔵 **Blue buttons** = Your shifts with swap options available
                - ⚪ **Gray buttons** = Your shifts with no swap options  
                - 🟢 **Green buttons** = Available swap dates (shown when you select a shift)
                - 🔷 **Dark blue buttons** = Shifts you're receiving (locked in)
                - 🟢 **Dark green buttons** = Shifts you're giving away (locked in)
                - **Gray text** = Regular days with no shifts
                """)
                
                # Add scrollable container with max height
                with st.container():
                    # Add a note about scrolling if many months
                    if months_to_show > 3:
                        st.info(f"📅 Showing {months_to_show} months of calendar data. Scroll down to see all months.")
                    
                    # Show date range being displayed
                    if all_dates:
                        st.write(f"**Date Range:** {min_shift_date} to {max_shift_date}")
                    else:
                        st.write(f"**Starting from:** {current_date}")
                    # Create a simple calendar grid
                    for month_offset in range(months_to_show):
                        display_date = current_date + timedelta(days=30 * month_offset)
                        year = display_date.year
                        month = display_date.month
                        
                        st.write(f"### {calendar.month_name[month]} {year}")
                        
                        # Create calendar grid
                        cal = calendar.monthcalendar(year, month)
                        
                        # Create columns for each day of week
                        col_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                        cols = st.columns(7)
                        
                        # Header row
                        for i, col_name in enumerate(col_names):
                            with cols[i]:
                                st.write(f"**{col_name}**")
                        
                        # Calendar rows
                        for week in cal:
                            cols = st.columns(7)
                            for i, day in enumerate(week):
                                with cols[i]:
                                    if day == 0:
                                        st.write("")
                                    else:
                                        date_obj = datetime(year, month, day).date()
                                        
                                        # Create unique key by including month and year
                                        unique_key = f"cal_{year}_{month:02d}_{day:02d}"
                                        
                                        if date_obj in shifts_by_date:
                                            # This date has one or more shifts
                                            shifts_on_date = shifts_by_date[date_obj]
                                            
                                            # Create a container for multiple shift buttons
                                            st.write(f"**{day}**")
                                            
                                            for shift_idx, shift in enumerate(shifts_on_date):
                                                # Create unique key for each shift
                                                shift_key = f"{unique_key}_shift_{shift_idx}"
                                                shift_count = len(shift.get('compatible_swaps', []))
                                                
                                                # Check if this specific shift is affected by locked swaps
                                                is_affected, affected_swap = is_date_affected_by_swap(date_obj, st.session_state.locked_swaps)
                                                
                                                # Create display text for this shift
                                                person_name = shift.get('person', 'Unknown')
                                                display_text = f"{shift['shift_type']} ({person_name})"
                                                
                                                if is_affected:
                                                    # This shift is part of a locked swap
                                                    if date_obj == affected_swap['your_date']:
                                                        # This is your original shift that you're giving away
                                                        st.markdown(f"""
                                                        <div class="dark-green-button" style="padding: 6px; border-radius: 4px; text-align: center; margin: 1px; width: 100%; cursor: pointer;" onclick="document.getElementById('{shift_key}').click()">
                                                            <strong>{display_text}</strong><br>
                                                            🔄 Swapping
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                        if st.button("Select", key=shift_key, help=f"Giving away shift on {date_obj}", use_container_width=True):
                                                            st.session_state.selected_date = date_obj
                                                            st.rerun()
                                                    else:
                                                        # This is a shift you're receiving
                                                        st.markdown(f"""
                                                        <div class="dark-blue-button" style="padding: 6px; border-radius: 4px; text-align: center; margin: 1px; width: 100%; cursor: pointer;" onclick="document.getElementById('{shift_key}').click()">
                                                            <strong>{display_text}</strong><br>
                                                            ✅ Receiving
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                        if st.button("Select", key=shift_key, help=f"Receiving shift on {date_obj}", use_container_width=True):
                                                            st.session_state.selected_date = date_obj
                                                            st.rerun()
                                                elif shift_count > 0:
                                                    # Blue button for shifts with swaps
                                                    # Check if this shift is currently selected
                                                    is_selected = st.session_state.selected_date == date_obj
                                                    
                                                    if is_selected:
                                                        # Darker blue for selected shift
                                                        st.markdown(f"""
                                                        <div class="dark-blue-button" style="padding: 6px; border-radius: 4px; text-align: center; margin: 1px; width: 100%; cursor: pointer;" onclick="document.getElementById('{shift_key}').click()">
                                                            <strong>{display_text}</strong><br>
                                                            {shift_count} swaps
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                        if st.button("Selected", key=shift_key, help=f"Currently selected shift on {date_obj}", use_container_width=True):
                                                            st.session_state.selected_date = None  # Deselect
                                                            st.rerun()
                                                    else:
                                                        # Regular blue for unselected shift
                                                        st.markdown(f"""
                                                        <div class="blue-button" style="padding: 6px; border-radius: 4px; text-align: center; margin: 1px; width: 100%; cursor: pointer;" onclick="document.getElementById('{shift_key}').click()">
                                                            <strong>{display_text}</strong><br>
                                                            {shift_count} swaps
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                        if st.button("Select", key=shift_key, help=f"Click to see swap options for {date_obj}", use_container_width=True):
                                                            st.session_state.selected_date = date_obj
                                                            st.rerun()
                                                else:
                                                    # Gray button for shifts without swaps
                                                    # Check if this shift is currently selected
                                                    is_selected = st.session_state.selected_date == date_obj
                                                    
                                                    if is_selected:
                                                        # Darker gray for selected shift
                                                        st.markdown(f"""
                                                        <div class="dark-green-button" style="padding: 6px; border-radius: 4px; text-align: center; margin: 1px; width: 100%; cursor: pointer;" onclick="document.getElementById('{shift_key}').click()">
                                                            <strong>{display_text}</strong><br>
                                                            No swaps
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                        if st.button("Selected", key=shift_key, help=f"Currently selected shift on {date_obj}", use_container_width=True):
                                                            st.session_state.selected_date = None  # Deselect
                                                            st.rerun()
                                                    else:
                                                        # Regular gray for unselected shift
                                                        st.markdown(f"""
                                                        <div class="gray-button" style="padding: 6px; border-radius: 4px; text-align: center; margin: 1px; width: 100%; cursor: pointer;" onclick="document.getElementById('{shift_key}').click()">
                                                            <strong>{display_text}</strong><br>
                                                            No swaps
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                        if st.button("Select", key=shift_key, help=f"No swap options for {date_obj}", use_container_width=True):
                                                            st.session_state.selected_date = date_obj
                                                            st.rerun()
                                        else:
                                            # Check if this date is affected by locked swaps (shift you're receiving)
                                            is_affected, affected_swap = is_date_affected_by_swap(date_obj, st.session_state.locked_swaps)
                                            
                                            if is_affected and date_obj == affected_swap['their_date']:
                                                # This is a shift you're receiving through a locked swap
                                                shift_type = affected_swap['their_shift'].split(' ', 1)[1]
                                                st.markdown(f"""
                                                <div class="dark-blue-button" style="padding: 8px; border-radius: 4px; text-align: center; margin: 2px; width: 100%; cursor: pointer;" onclick="document.getElementById('{unique_key}').click()">
                                                    <strong>{day}</strong><br>
                                                    {shift_type}<br>
                                                    ✅ Receiving
                                                </div>
                                                """, unsafe_allow_html=True)
                                                if st.button("Select", key=unique_key, help=f"Receiving shift from locked swap - {date_obj}", use_container_width=True):
                                                    st.session_state.selected_date = date_obj
                                                    st.rerun()
                                            else:
                                                # Check if this date is a swap option for any selected shift
                                                swap_options = []
                                                if st.session_state.selected_date and st.session_state.selected_date in shifts_by_date:
                                                    selected_shifts = shifts_by_date[st.session_state.selected_date]
                                                    for selected_shift in selected_shifts:
                                                        for swap in selected_shift.get('compatible_swaps', []):
                                                            if swap['target_date'] == date_obj:
                                                                swap_options.append(swap)
                                                
                                                if swap_options:
                                                    # Show multiple swap options
                                                    st.write(f"**{day}** - Swap Options")
                                                    
                                                    for swap_idx, swap_details in enumerate(swap_options):
                                                        swap_key = f"{unique_key}_swap_{swap_idx}"
                                                        
                                                        st.markdown(f"""
                                                        <div class="green-button" style="padding: 6px; border-radius: 4px; text-align: center; margin: 1px; width: 100%; cursor: pointer;" onclick="document.getElementById('{swap_key}').click()">
                                                            <strong>{swap_details['target_person']}</strong><br>
                                                            {swap_details['target_shift_type']}
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                        
                                                        if st.button("Select", key=swap_key, help=f"Swap with {swap_details['target_person']}", use_container_width=True):
                                                            # Add this swap to current plan
                                                            if st.session_state.selected_date and st.session_state.selected_date in shifts_by_date:
                                                                selected_shifts = shifts_by_date[st.session_state.selected_date]
                                                                
                                                                # Find the specific swap details
                                                                for selected_shift in selected_shifts:
                                                                    for swap in selected_shift.get('compatible_swaps', []):
                                                                        if swap['target_date'] == date_obj and swap['target_person'] == swap_details['target_person']:
                                                                            # Create swap entry
                                                                            swap_entry = {
                                                                                'your_shift': f"{st.session_state.selected_date.strftime('%Y-%m-%d')} {selected_shift['shift_type']}",
                                                                                'their_shift': f"{date_obj.strftime('%Y-%m-%d')} {swap['target_shift_type']}",
                                                                                'target_person': swap['target_person'],
                                                                                'your_date': st.session_state.selected_date,
                                                                                'their_date': date_obj
                                                                            }
                                                                            
                                                                            # Check if this swap is already in the plan
                                                                            already_exists = any(
                                                                                s['your_date'] == swap_entry['your_date'] and 
                                                                                s['their_date'] == swap_entry['their_date'] and
                                                                                s['target_person'] == swap_entry['target_person']
                                                                                for s in st.session_state.current_swap_plan
                                                                            )
                                                                            
                                                                            if not already_exists:
                                                                                st.session_state.current_swap_plan.append(swap_entry)
                                                                                st.success(f"Added swap: {swap_entry['your_shift']} ↔ {swap_entry['their_shift']}")
                                                                            else:
                                                                                st.warning("This swap is already in your plan!")
                                                                            break
                                                                    if not already_exists:
                                                                        break
                                                            st.rerun()
                                                else:
                                                    # Regular day
                                                    st.write(f"{day}")
                        
                        st.write("---")  # Separator between months
                
                # Swap Planning Section
                st.subheader("📋 Swap Planning")
                
                # Show current swap plan
                if st.session_state.current_swap_plan:
                    st.write("**Your Current Swap Plan:**")
                    for i, swap in enumerate(st.session_state.current_swap_plan):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"• {swap['your_shift']} ↔ {swap['their_shift']}")
                        with col2:
                            if st.button("Remove", key=f"remove_{i}"):
                                st.session_state.current_swap_plan.pop(i)
                                st.rerun()
                        with col3:
                            if st.button("Lock In", key=f"lock_{i}"):
                                st.session_state.locked_swaps.append(swap)
                                st.session_state.current_swap_plan.pop(i)
                                st.rerun()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Clear All", type="secondary"):
                            st.session_state.current_swap_plan = []
                            st.rerun()
                    with col2:
                        if st.button("Lock In All", type="primary"):
                            # Move all current swaps to locked swaps
                            st.session_state.locked_swaps.extend(st.session_state.current_swap_plan)
                            st.session_state.current_swap_plan = []
                            st.rerun()
                else:
                    st.info("No swaps in your current plan. Click on green 'Swap Option' dates to add them.")
                
                # Show locked swaps with history
                if st.session_state.locked_swaps:
                    st.write("**Locked In Swaps (3-Way Enabled):**")
                    for i, swap in enumerate(st.session_state.locked_swaps):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.success(f"✅ {swap['your_shift']} ↔ {swap['their_shift']} (with {swap['target_person']})")
                        with col2:
                            if st.button("Unlock", key=f"unlock_{i}"):
                                st.session_state.locked_swaps.pop(i)
                                st.rerun()
                    
                    st.info("💡 **3-Way Swaps Enabled**: Your received shifts are now available for additional swaps!")
                
                
                # Summary
                total_swaps = sum(len(shift['compatible_swaps']) for _, shift in calendar_df.iterrows())
                shifts_with_swaps = sum(1 for _, shift in calendar_df.iterrows() if len(shift['compatible_swaps']) > 0)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Potential Swaps", total_swaps)
                with col2:
                    st.metric("Shifts with Swap Options", shifts_with_swaps)
                with col3:
                    st.metric("Planned Swaps", len(st.session_state.current_swap_plan))
                with col4:
                    st.metric("Locked Swaps", len(st.session_state.locked_swaps))
                
                # Export functionality
                if st.session_state.locked_swaps:
                    st.subheader("📤 Export Swap Plan")
                    
                    # Create export data
                    export_data = []
                    for i, swap in enumerate(st.session_state.locked_swaps):
                        export_data.append({
                            'Swap #': i + 1,
                            'Your Shift': swap['your_shift'],
                            'Their Shift': swap['their_shift'],
                            'Swap Partner': swap['target_person']
                        })
                    
                    if export_data:
                        df_export = pd.DataFrame(export_data)
                        csv = df_export.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Swap Plan as CSV",
                            data=csv,
                            file_name=f"swap_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
            else:
                st.info("No potential swaps found with the selected criteria")
        
        else:
            st.info("👆 Please select yourself and potential swap partners in the sidebar")

if __name__ == "__main__":
    main()
