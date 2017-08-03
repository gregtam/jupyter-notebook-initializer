import threading

from IPython.core.magic import register_cell_magic
import numpy as np
import pandas as pd

class ThreadManager:
    def __call__(self, num_rows=5):
        """Returns the num_rows most recent threads (Default: 5)"""
        return self.thread_df.tail(num_rows).iloc[::-1]
    
    def __init__(self):
        df_cols = ['start_time', 'finish_time', 'exec_time',
                   'cell_text', 'comment']
        self.thread_df = pd.DataFrame(columns=df_cols)
        
    def add_thread(self, cell_text, comment='N/A'):
        thread_id = self.get_next_thread_id()
        start_time = datetime.now()
        self.thread_df.loc[thread_id] = [start_time, '', '', cell_text, comment]
        print 'Started Thread {} at {}.\nComment: {}'.format(thread_id, start_time, comment)
        
    def finish_thread(self, thread_id):
        if self.thread_df.loc[thread_id, 'finish_time'] == '':
            # Set finish time
            finish_time = datetime.now()
            self.thread_df.loc[thread_id, 'finish_time'] = finish_time
            
            # Set execution time
            exec_time = finish_time - self.thread_df.loc[thread_id, 'start_time']
            self.thread_df.loc[thread_id, 'exec_time'] = exec_time
            
            # Print comment
            comment = str(self.thread_df.loc[thread_id, 'comment'])
            print 'Finished Thread {} at {}.\nDone in {}.\nComment: {}'.format(thread_id, finish_time, exec_time, comment)
                        
        else:
            raise Exception('Cannot finish an already completed thread.')
        
    def raise_thread_error(self, thread_id, error_message):
        if self.thread_df.loc[thread_id, 'finish_time'] == '':
            self.thread_df.loc[thread_id, 'finish_time'] = 'Exception: {}'.format(error_message)
            self.thread_df.loc[thread_id, 'exec_time'] = 'Exception: {}'.format(error_message)
        
    def get_next_thread_id(self):
        return self.thread_df.shape[0]
    
    def get_finished_threads(self):
        return self.thread_df[self.thread_df['finish_time'] != '']

    def get_unfinished_threads(self):
        return self.thread_df[self.thread_df['finish_time'] == '']
    
@register_cell_magic
def background(line, cell):
    """
    Runs whatever is in the cell in a separate thread.
    This allows the user to run cells in the background
    so that additional cells can be run concurrently. This
    will also micromanage by labelling each thread with an
    id number.
    
    Whatever follows after specifying '%%background' will
    be used as a comment to label the process if the id
    number is not descriptive enough.
    """
    def run_cell(cell_value):
        thread_id = thread_manager.get_next_thread_id()
        if len(line) > 0:
            thread_manager.add_thread(cell_text=cell, comment=line)
        else:
            thread_manager.add_thread(cell_text=cell)
            
        try:
            exec cell_value in globals()
        except Exception as error_message:
            thread_manager.raise_thread_error(thread_id, error_message)
            raise Exception(error_message)
            
        thread_manager.finish_thread(thread_id)
        
    thread = threading.Thread(target=run_cell, args=(cell, ))
    thread.start()

# We delete this to avoid name conflicts for automagic to work
del background
