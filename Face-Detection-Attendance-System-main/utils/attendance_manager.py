"""
Attendance Management System
"""

import sqlite3
import pandas as pd
import csv
from datetime import datetime, timedelta
from pathlib import Path
from config import ATTENDANCE_CONFIG, FACE_NAME_MAPPING
from utils.logger import logger

class AttendanceManager:
    """
    Manages attendance tracking and reporting
    """
    
    def __init__(self, db_path=None):
        self.db_path = db_path or ATTENDANCE_CONFIG["db_path"]
        self.attendance_file = ATTENDANCE_CONFIG["attendance_file"]
        self.daily_report_dir = Path(ATTENDANCE_CONFIG["daily_report_file"]).parent
        self.monthly_report_dir = Path(ATTENDANCE_CONFIG["monthly_report_file"]).parent
        
        # Create directories
        self.daily_report_dir.mkdir(exist_ok=True)
        self.monthly_report_dir.mkdir(exist_ok=True)
        
        self._init_database()
        logger.log_system_event("AttendanceManager initialized")
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create attendance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    person_name TEXT,
                    check_in_time DATETIME,
                    check_out_time DATETIME,
                    total_hours REAL,
                    status TEXT,
                    confidence REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create persons table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE,
                    face_id INTEGER UNIQUE,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.log_system_event("Database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def add_person(self, name, face_id):
        """
        Add a new person to the database
        
        Args:
            name: Person's name
            face_id: Face ID from recognition system
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO persons (name, face_id)
                VALUES (?, ?)
            ''', (name, face_id))
            
            conn.commit()
            conn.close()
            
            logger.log_system_event("Person added", f"Name: {name}, Face ID: {face_id}")
            
        except Exception as e:
            logger.error(f"Failed to add person: {e}")
    
    def record_attendance(self, face_id, action="check_in", confidence=None):
        """
        Record attendance for a person
        
        Args:
            face_id: Face ID from recognition
            action: "check_in" or "check_out"
            confidence: Recognition confidence score
        """
        try:
            person_name = FACE_NAME_MAPPING.get(face_id, f"Unknown_{face_id}")
            current_time = datetime.now()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if action == "check_in":
                # Check if already checked in today
                cursor.execute('''
                    SELECT id FROM attendance 
                    WHERE person_id = ? AND DATE(check_in_time) = DATE(?)
                    AND check_out_time IS NULL
                ''', (face_id, current_time))
                
                if cursor.fetchone():
                    logger.warning(f"Person {person_name} already checked in today")
                    conn.close()
                    return False
                
                # Record check-in
                cursor.execute('''
                    INSERT INTO attendance (person_id, person_name, check_in_time, confidence, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (face_id, person_name, current_time, confidence, "checked_in"))
                
                logger.log_attendance(person_name, "CHECK_IN", confidence)
                
            elif action == "check_out":
                # Find today's check-in record
                cursor.execute('''
                    SELECT id, check_in_time FROM attendance 
                    WHERE person_id = ? AND DATE(check_in_time) = DATE(?)
                    AND check_out_time IS NULL
                ''', (face_id, current_time))
                
                record = cursor.fetchone()
                if not record:
                    logger.warning(f"No check-in record found for {person_name}")
                    conn.close()
                    return False
                
                # Calculate total hours
                check_in_time = datetime.fromisoformat(record[1])
                total_hours = (current_time - check_in_time).total_seconds() / 3600
                
                # Update check-out
                cursor.execute('''
                    UPDATE attendance 
                    SET check_out_time = ?, total_hours = ?, status = ?
                    WHERE id = ?
                ''', (current_time, total_hours, "checked_out", record[0]))
                
                logger.log_attendance(person_name, "CHECK_OUT", confidence)
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to record attendance: {e}")
            return False
    
    def get_today_attendance(self):
        """
        Get today's attendance records
        
        Returns:
            DataFrame with today's attendance
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    person_name,
                    check_in_time,
                    check_out_time,
                    total_hours,
                    status,
                    confidence
                FROM attendance 
                WHERE DATE(check_in_time) = DATE('now')
                ORDER BY check_in_time
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get today's attendance: {e}")
            return pd.DataFrame()
    
    def get_attendance_summary(self, start_date=None, end_date=None):
        """
        Get attendance summary for a date range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with attendance summary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if start_date is None:
                start_date = datetime.now().strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get total attendance
            cursor.execute('''
                SELECT COUNT(DISTINCT person_id) 
                FROM attendance 
                WHERE DATE(check_in_time) BETWEEN ? AND ?
            ''', (start_date, end_date))
            
            total_attended = cursor.fetchone()[0]
            
            # Get total persons
            cursor.execute('SELECT COUNT(*) FROM persons WHERE is_active = 1')
            total_persons = cursor.fetchone()[0]
            
            # Get average hours
            cursor.execute('''
                SELECT AVG(total_hours) 
                FROM attendance 
                WHERE DATE(check_in_time) BETWEEN ? AND ?
                AND total_hours IS NOT NULL
            ''', (start_date, end_date))
            
            avg_hours = cursor.fetchone()[0] or 0
            
            conn.close()
            
            summary = {
                'total_persons': total_persons,
                'total_attended': total_attended,
                'attendance_rate': (total_attended / total_persons * 100) if total_persons > 0 else 0,
                'average_hours': round(avg_hours, 2),
                'start_date': start_date,
                'end_date': end_date
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get attendance summary: {e}")
            return {}
    
    def generate_daily_report(self, date=None):
        """
        Generate daily attendance report
        
        Args:
            date: Date for report (YYYY-MM-DD), defaults to today
        """
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            # Get attendance data
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    person_name,
                    check_in_time,
                    check_out_time,
                    total_hours,
                    status,
                    confidence
                FROM attendance 
                WHERE DATE(check_in_time) = ?
                ORDER BY check_in_time
            '''
            
            df = pd.read_sql_query(query, conn, params=(date,))
            conn.close()
            
            if df.empty:
                logger.warning(f"No attendance data for {date}")
                return
            
            # Generate report
            report_file = self.daily_report_dir / f"daily_report_{date}.csv"
            df.to_csv(report_file, index=False)
            
            # Generate summary
            summary = self.get_attendance_summary(date, date)
            
            summary_file = self.daily_report_dir / f"daily_summary_{date}.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Daily Attendance Report - {date}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Persons: {summary.get('total_persons', 0)}\n")
                f.write(f"Total Attended: {summary.get('total_attended', 0)}\n")
                f.write(f"Attendance Rate: {summary.get('attendance_rate', 0):.1f}%\n")
                f.write(f"Average Hours: {summary.get('average_hours', 0):.2f}\n\n")
                f.write("Detailed Records:\n")
                f.write("-" * 30 + "\n")
                
                for _, row in df.iterrows():
                    f.write(f"{row['person_name']}: {row['check_in_time']} - {row['check_out_time'] or 'Not checked out'}\n")
            
            logger.log_system_event("Daily report generated", f"Date: {date}")
            
        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
    
    def generate_monthly_report(self, year=None, month=None):
        """
        Generate monthly attendance report
        
        Args:
            year: Year for report
            month: Month for report
        """
        try:
            if year is None or month is None:
                now = datetime.now()
                year = now.year
                month = now.month
            
            start_date = f"{year:04d}-{month:02d}-01"
            end_date = (datetime(year, month, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            end_date = end_date.strftime('%Y-%m-%d')
            
            # Get monthly data
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    person_name,
                    DATE(check_in_time) as date,
                    check_in_time,
                    check_out_time,
                    total_hours,
                    status
                FROM attendance 
                WHERE DATE(check_in_time) BETWEEN ? AND ?
                ORDER BY person_name, check_in_time
            '''
            
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()
            
            if df.empty:
                logger.warning(f"No attendance data for {year}-{month:02d}")
                return
            
            # Generate report
            report_file = self.monthly_report_dir / f"monthly_report_{year}_{month:02d}.csv"
            df.to_csv(report_file, index=False)
            
            # Generate summary
            summary = self.get_attendance_summary(start_date, end_date)
            
            summary_file = self.monthly_report_dir / f"monthly_summary_{year}_{month:02d}.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Monthly Attendance Report - {year}-{month:02d}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Persons: {summary.get('total_persons', 0)}\n")
                f.write(f"Total Attended: {summary.get('total_attended', 0)}\n")
                f.write(f"Attendance Rate: {summary.get('attendance_rate', 0):.1f}%\n")
                f.write(f"Average Hours: {summary.get('average_hours', 0):.2f}\n\n")
                
                # Daily breakdown
                daily_stats = df.groupby('date').agg({
                    'person_name': 'count',
                    'total_hours': 'mean'
                }).rename(columns={'person_name': 'attended_count'})
                
                f.write("Daily Breakdown:\n")
                f.write("-" * 30 + "\n")
                for date, row in daily_stats.iterrows():
                    f.write(f"{date}: {row['attended_count']} persons, {row['total_hours']:.2f} avg hours\n")
            
            logger.log_system_event("Monthly report generated", f"Year: {year}, Month: {month}")
            
        except Exception as e:
            logger.error(f"Failed to generate monthly report: {e}")
    
    def get_late_attendance(self, date=None, threshold_minutes=15):
        """
        Get list of late attendees
        
        Args:
            date: Date to check (YYYY-MM-DD)
            threshold_minutes: Minutes after check-in time to consider late
            
        Returns:
            DataFrame with late attendance records
        """
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            check_in_time = datetime.strptime(ATTENDANCE_CONFIG["check_in_time"], "%H:%M")
            late_threshold = check_in_time + timedelta(minutes=threshold_minutes)
            
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    person_name,
                    check_in_time,
                    status,
                    confidence
                FROM attendance 
                WHERE DATE(check_in_time) = ?
                AND TIME(check_in_time) > TIME(?)
                ORDER BY check_in_time
            '''
            
            df = pd.read_sql_query(query, conn, params=(date, late_threshold.strftime('%H:%M')))
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get late attendance: {e}")
            return pd.DataFrame()
    
    def export_attendance_data(self, start_date=None, end_date=None, output_file=None):
        """
        Export attendance data to CSV
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_file: Output file path
        """
        try:
            if start_date is None:
                start_date = datetime.now().strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if output_file is None:
                output_file = f"attendance_export_{start_date}_to_{end_date}.csv"
            
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    person_name,
                    check_in_time,
                    check_out_time,
                    total_hours,
                    status,
                    confidence,
                    created_at
                FROM attendance 
                WHERE DATE(check_in_time) BETWEEN ? AND ?
                ORDER BY check_in_time
            '''
            
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()
            
            df.to_csv(output_file, index=False)
            logger.log_system_event("Attendance data exported", f"File: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to export attendance data: {e}")
            return None 