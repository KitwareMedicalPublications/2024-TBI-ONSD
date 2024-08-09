''' ONSD Annotator UI

This Ttk-based script creates a convenient UI for executing ImageViewer over a research study folder
and allows for an annotator to be selected.
'''

import ttkbootstrap as ttk
from ttkbootstrap.tooltip import ToolTip
import tkinter.filedialog as fd
from ttkbootstrap.constants import *
from abc import ABC, abstractmethod
import os
from pathlib import Path
import json
from glob import glob
import re
import numpy as np
from threading import Thread
import subprocess
import random
from ttkwidgets import checkboxtreeview
import pandas as pd

IMAGEVIEWER_PATH = Path(__file__).parent / 'ImageViewer' / 'ImageViewer'

_PROCESS = None

class Configuration:
    '''
    Handles default and user configuration values and file IO.
    '''

    def __init__(self):
        '''
        Loads default and user configuration data.  If the user configuration data doesn't exist
        populates it with default values and creates the user configuration file.

        Default data is stored in the application directory, e.g., __file__/../default_config.json and
        user data is stored under USER_HOME/.annotator/config.json
        '''
        self.data = dict()
        self.path = Path.home() / '.annotator' / 'config.json'

        self.default_path = Path(__file__).parent / 'default_config.json'

        with open(str(self.default_path), 'r') as fp:
            self.default_data = json.load(fp)

        if not self.path.exists():
            os.makedirs(self.path.parent, exist_ok=True)
            self.data['default_folder'] = str(Path.home())
            self.data['default_users'] = self.default_data['users']
            self.data['default_user'] = self.default_data['users'][0] if len(self.default_data['users']) > 0 else None
            self.data['default_mode'] = 'Update'
            self.data['default_replicate_id'] = 1
            self.save()
        else:
            self.load()

    def push(self, key, value):
        '''
        Assigns self.data[key] to value and saves to the user config file

        Parameters
        ----------
        key : str
        value : object
        '''
        self.data[key] = value
        self.save()

    def save(self):
        '''
        Writes self.data to user configuration file (e.g., USER_HOME/.annotator/config.json)
        '''
        with open(str(self.path), 'w') as fp:
            json.dump(self.data, fp)

    def load(self):
        '''
        Loads self.data from user configuration file (e.g., USER_HOME/.annotator/config.json)
        '''
        with open(str(self.path), 'r') as fp:
            self.data = json.load(fp)

class MainFrame(ttk.Frame):
    def __init__(self, root, config):
        super().__init__(root)
        self.todo = []
        self.config = config
        self.process_thread = None
        self.grid_rowconfigure(3, weight=1, minsize=800)
        self.grid_columnconfigure(1, weight=1, minsize=800)
        
        lbl1 = ttk.Label(self, text='Base Directory:')
        lbl1.grid(row=0, column=0, sticky=W, padx=5, pady=10)

        self.dir_label_value = ttk.StringVar(root, config.data['default_folder'])
       
        self.dir_label = ttk.Label(self, textvariable=self.dir_label_value)
        self.dir_label.grid(row=0, column=1)
        
        self.dir_label_tooltip = None
        # will open a file dialog
        self.dir_button = ttk.Button(self, text='Select Folder...')
        self.dir_button.grid(row=0, column=2, sticky=E, padx=10, pady=5)
        ToolTip(self.dir_button, text='Select the base research directory, i.e., the parent directory containing \'preprocessed\' and \'raw\'')

        self.user_label = ttk.Label(self, text='User:')
        self.user_label.grid(row=1, column=0, sticky=W, padx=5, pady=10)
        
        # TODO make an add user button, interrogate possible users from config data as well as directory structure

        self.user_combo = ttk.Combobox(self, values=self.config.default_data['users'])
        self.user_combo.set(self.config.data['default_user'])
        self.user_combo.grid(row=1, column=1, sticky='ew')

        self.user_button = ttk.Button(self, text='Add User...')
        self.user_button.grid(row=1, column=2, sticky=EW, padx=10, pady=5)

        replicate_label = ttk.Label(self, text='Replicate:')
        replicate_label.grid(row=2, column=0, sticky=W, padx=5, pady=10)
        
        self.replicate_combo = ttk.Combobox(self, values=[''])
        self.replicate_combo.set('')
        self.replicate_combo.grid(row=2, column=1, sticky='ew')
        
        self.replicate_button = ttk.Button(self, text='New Replicate')
        self.replicate_button.grid(row=2, column=2, sticky=EW, padx=10, pady=5)
        
        # we want this to be general, so we don't actually know what fieldN actually mean
        # literally just subdir names if they are available (relative to preprocessed dir)
        self.items_treeview = checkboxtreeview.CheckboxTreeview(self, columns=('field1', 'field2', 'annotated', 'filename'))
        self.items_treeview.heading('field1', text='Field 1')
        self.items_treeview.heading('field2', text='Field 2')
        self.items_treeview.heading('annotated', text='Annotated?')
        self.items_treeview.heading('filename', text='Filename')

        self.items_treeview.insert('', 0, values=['One', 'two', 'three'])
        self.items_treeview.grid(row=3, column=0, sticky=NSEW, columnspan=2, padx=5, pady=10)

        verscrlbar = ttk.Scrollbar(self, orient ="vertical", command = self.items_treeview.yview)
        verscrlbar.grid(row=3, column=2, sticky='NSW')
        self.items_treeview.configure(xscrollcommand = verscrlbar.set)

        shuffle_var = ttk.IntVar(value=0)
        self.shuffle_check = ttk.Checkbutton(self, text='Shuffle', state='selected')
        self.shuffle_check.state(['!alternate'])
        self.shuffle_check.state(['selected'])
        self.shuffle_check.grid(row=4, column=0, sticky=E, padx=5, pady=10)
        # print(self.shuffle_check.getint())
        self.process_label = ttk.Label(self, text='')
        self.process_label.grid(row=4, column=1, stick=E, padx=5, pady=10)

        self.process_button = ttk.Button(self, text='Start')
        self.process_button.grid(row=4, column=2, sticky=EW, padx=10, pady=10)

        self.treeview_context = ttk.Menu(self, tearoff=0)
        self.treeview_context.add_command(label='Check', command=self.context_check)
        self.treeview_context.add_command(label='Uncheck', command=self.context_uncheck)
        self.treeview_context.add_command(label='Select All', command=self.context_select_all)
        self.treeview_context.add_command(label='Select None', command=self.context_select_none)  
        self.items_treeview.bind('<Button-3>', self.context_popup)
        self.items_treeview.bind('<Button-1>', self.treeview_checkchange, add=True)
        
        
        self.current_state = PresentationModel(self)
        self.current_state.pass_state(None)

    def treeview_checkchange(self, event):
        self.current_state.update_process(True)

    def context_popup(self, event):
        try:
            self.treeview_context.tk_popup(event.x_root, event.y_root)
        finally:
            self.treeview_context.grab_release()

    def context_check(self):
        for i in self.items_treeview.selection():
            self.items_treeview.change_state(i, 'checked')
        self.treeview_checkchange(None)

    def context_uncheck(self):
        for i in self.items_treeview.selection():
            self.items_treeview.change_state(i, 'unchecked')
        self.treeview_checkchange(None)

    def context_select_all(self):
        self.items_treeview.selection_add(self.items_treeview.get_children())

    def context_select_none(self):
        self.items_treeview.selection_remove(self.items_treeview.get_children())

def run_image_viewer(todo):
    '''
    Executes an ImageViewer subprocess per item in todo and continues once ImageViewer is closed

    Parameters
    ----------
    todo : list of (str, str, bool)
        Each entry of todo is the preprocessed file path, the prefix path for ImageViewer output, and bool on whether it has already been processed and should be ignored
        Note, todo is shuffled when this is called
    '''

    for f, out_f, complete in todo:
        if not complete:
            os.makedirs(str(Path(out_f).parent), exist_ok=True)
            global _PROCESS
            # command args, only move by one slice, start at slice 0, use prefix out_f for output files, start in ONSD ruler mode
            _PROCESS = subprocess.Popen([str(IMAGEVIEWER_PATH), '--fixedSliceDelta', '1', '-S', out_f, '-s', '0', '-W', 'r,o,r,t,sheathe', f])
            _PROCESS.wait()
            print(_PROCESS, flush=True)
            returncode = _PROCESS.returncode
            _PROCESS = None
            rulers_p = Path(out_f + '.rulers.json')

            # if no rulers are selected by the user, ImageViewer appears to not output anything
            # so, we'll create an "empty" json
            # we check returncode here because if the user simply closes the Annotator app, i.e., to stop in the 
            # middle of annotating, we don't want to save anything
            if not rulers_p.exists() and returncode == 0:
                with open(str(rulers_p), 'w') as fp:
                    json.dump(dict(), fp)

class PresentationModel:
    '''
    Initial state of the application.  TODO: this needs to be refactored to pull out the Domain Model
    '''
    def __init__(self, mainframe):
        self.mainframe = mainframe
    
    def pass_state(self, prev_state):
        if prev_state is None: # initializing
            self.mainframe.dir_button.bind('<ButtonRelease>', self.hnd_button_release, add='')
            self.mainframe.dir_label_value.trace('w', self.hnd_dir_label_value)
            self.hnd_dir_label_value() # verify the default value of the dir_label_value
            self.mainframe.replicate_button.bind('<ButtonRelease>', self.hnd_replicate_button, add='')
            self.mainframe.process_button.bind('<ButtonRelease>', self.hnd_process_button, add='')
            self.mainframe.user_button.bind('<ButtonRelease>', self.hnd_user_button, add='')
            self.mainframe.user_combo.bind('<<ComboboxSelected>>', self.hnd_user_combo)   
            self.mainframe.replicate_combo.bind('<<ComboboxSelected>>', self.hnd_replicate_combo)  
    
    def hnd_user_combo(self, *args):
        if (not self.mainframe.instate(['disabled'])):
            self.update_replicate_list()
            self.update_todo()
            self.update_process(True)

    def hnd_replicate_combo(self, *args):
        if (not self.mainframe.instate(['disabled'])):
            self.update_todo()
            self.update_process(True)

    def hnd_button_release(self, *args):
        f = fd.askdirectory(mustexist=True)
        self.mainframe.dir_label_value.set(f)

    def hnd_user_button(self, *args):
        new_user = ttk.dialogs.Querybox.get_string(prompt='Enter username (no hyphens)')
        if new_user is not None:
            values = list(self.mainframe.user_combo['values'])
            values.append(new_user)
            self.mainframe.user_combo['values'] = values
            self.mainframe.user_combo.set(new_user)
            self.mainframe.user_combo.event_generate('<<ComboboxSelected>>')

    def hnd_dir_label_value(self, *args):
        '''
        Parameters
        ----------
        *args : tuple
            Throwaway args passed by callback.  Ttk has a poorly documented tuple of strings that are not useful.

        '''
        s = self.mainframe.dir_label_value.get()
        p = Path(s)

        if not (p / 'preprocessed').exists():
            # update file list and replicate
            self.mainframe.replicate_button.state(['disabled'])
            self.mainframe.todo = []
            self.update_process(False)
            
            # if not none, do nothing as we already have a warning
            if self.mainframe.dir_label_tooltip is None:
                self.mainframe.dir_label_tooltip = ToolTip(self.mainframe.dir_label, text='Selected directory must exist and have a \'preprocessed\' subdirectory')
                self.mainframe.dir_label.configure(bootstyle=DANGER)
                
        else:
            self.update_replicate_list()
            self.mainframe.replicate_button.state(['!disabled'])
            self.update_todo()
            self.update_process(True)
            if self.mainframe.dir_label_tooltip is not None:
                self.mainframe.dir_label_tooltip.hide_tip()
                self.mainframe.dir_label_tooltip = None
                self.mainframe.dir_label.configure(bootstyle=INFO)

    def hnd_replicate_button(self, *args):
        values = self.mainframe.replicate_combo['values']
        x = str(np.max([int(v) for v in values]) + 1)
        y = list(values)
        y.append(x)
        self.mainframe.replicate_combo['values'] = y
        self.mainframe.replicate_combo.set(x)
        self.mainframe.replicate_combo.event_generate('<<ComboboxSelected>>')

    def update_process(self, validate):
        if not validate:
            self.mainframe.process_label.configure(text='Select valid directory')
            self.mainframe.process_label.configure(bootstyle=DANGER)
            self.mainframe.process_button.state(['disabled'])
        else:
            n = len(self.mainframe.todo)
            n_todo = len([x for x, y, z in self.mainframe.todo if not z])
            n_selected = len([x for x in self.mainframe.items_treeview.get_checked() if self.mainframe.items_treeview.item(x)['values'][2] == 'False'])
            self.mainframe.process_label.configure(text=f'{n} files total.  {n_selected} (out of {n_todo}) annotations remaining.')
            self.mainframe.process_label.configure(bootstyle=INFO)
            self.mainframe.process_button.state(['!disabled'])

    def hnd_process_button(self, *args):
        todo = self.mainframe.todo.copy()

        

        # only run on items that are checked
        todo = [ todo[x] for x in range(len(todo)) if 'checked' in self.mainframe.items_treeview.item(self.todo_iids[x])['tags'] ]
        
        # shuffle if the option is selected
        if self.mainframe.shuffle_check.instate(['selected']):
            random.shuffle(todo)

        self.mainframe.process_thread = Thread(target=run_image_viewer, args=[todo], daemon=True)
        self.mainframe.process_thread.start()

    def update_replicate_list(self):
        '''
        Pulls a list of files matching the current selected user in the current research directory.  Updates
        the available replicate list and the list of files that have/need to be processed.
        '''
        # get replicates
        # get replicates by name
        cur_user = self.mainframe.user_combo.get()
        rep_regex = r'manual_onsd-' + cur_user + r'-(?P<replicate>[0-9]+)'
        dirs = glob(self.mainframe.dir_label_value.get() + '/manual_onsd*')

        # see if there are existing replicates
        replicates = []
        for d in dirs:
            print(d)
            m = re.search(rep_regex, d)
            if m is not None:
                replicates.append(m.group('replicate'))

        # add default replicate of '1' if none exist
        replicates = replicates if len(replicates) > 0 else ['1']
        tmp = [ int(r) for r in replicates ]

        # default to most recent replicate
        cur_r = str(np.max(tmp))

        self.mainframe.replicate_combo['values'] = replicates
        self.mainframe.replicate_combo.set(cur_r)

    def corresponding_onsd(self, f, research_dir, user, replicate):
        '''
        Given a preprocessed image or video file, return the file path corresponding to the manual
        onsd regardless of whether it exists.

        Parameters
        ----------
        f : str
            Path to preprocessed image/video file (e.g., '.mha' file)
        research_dir : str
            Path to parent directory of preprocessed
        user : str
            User string
        replicate : str
            Replicate ID

        Returns
        -------
        str
            Prefix for output ONSD measurements
        '''
        return f.replace('preprocessed', f'manual_onsd-{user}-{replicate}').replace('.mha', '')

    def find_files(self, research_dir):
        '''
        Find files to annotate.  This is a bit of hack, it would better if we gave options to load
        a config file or search a directory.  However, likely moving away from Annotator in the future.  If
        there is a file called "annotate-list.csv" in the research directory, then use that as a list of files.  If not,
        retrieve every .mha file in "RESEARCH_DIR/preprocessed" and its subdirs.

        Parameters
        ----------
        research_dir : str
            Search directory that contains ./preprocessed and possibly ./annotate-list.csv

        Returns
        -------
        list of str
    
        '''
        lst_file_path = Path(f'{research_dir}/annotate-list.csv')
        
        files = []
        if lst_file_path.exists():
            df = pd.read_csv(lst_file_path)
            files = list(df['file'])
        else:
            glob_str = f'{research_dir}/preprocessed/**/*.mha'
            files = glob(glob_str, recursive=True)
        return files

    def update_todo(self):
        '''
        Checks how many image/video files are in a directory, how many have been annotated, and updates
        the todo list on self.mainframe.todo.  Called when directory changes.  
        '''
        research_dir = self.mainframe.dir_label_value.get()
        user = self.mainframe.user_combo.get()
        replicate = self.mainframe.replicate_combo.get()

        pre_files = self.find_files(research_dir)

        self.mainframe.todo = []
        for f in pre_files:
            k = self.corresponding_onsd(f, research_dir, user, replicate)
            
            # check if ONSD has already been measured by file existence
            complete = False
            if Path(k + '.rulers.json').exists():
                complete = True
                
            self.mainframe.todo.append((f, k, complete))

        # clear current items
        self.mainframe.items_treeview.delete(*self.mainframe.items_treeview.get_children())
        
        def get_fields(f):
            ans = ['', '']
            r = Path(os.path.relpath(f, str(Path(research_dir) / 'preprocessed'))).parts
            for i in range(min(len(r), 2)):
                ans[i] = r[i]
            
            return ans

        self.todo_iids = []
        # add items
        for f, k, c in self.mainframe.todo:
            fields = get_fields(f)
            self.todo_iids.append(self.mainframe.items_treeview.insert('', 0, values=(fields[0], fields[1], c, Path(f).name)))



if __name__ == '__main__':
    root = ttk.Window(title='Annotator - Version 0.2', themename='superhero', minsize=(800,800))
    root.place_window_center()
    mf = MainFrame(root, Configuration())
    mf.grid(column=0, row=0, sticky=NSEW) # set the parent grid to fill window

    def on_closing():
        # save the configuration on closing so that the user has the same values when reloading the application
        mf.config.data['default_folder'] = mf.dir_label_value.get()
        mf.config.data['default_user'] = mf.user_combo.get()
        mf.config.data['default_replicate_id'] = mf.replicate_combo.get()
        mf.config.data['default_users'] = mf.user_combo['values']
        mf.config.save()

        if _PROCESS is not None:
            _PROCESS.terminate()

        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()