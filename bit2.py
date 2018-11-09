from time import gmtime, strftime, sleep, time
from ctypes import windll, c_uint, Structure, c_long, byref
from pywin32 import GetKeyState, GetAsyncKeyState
import mouse
import keyboard
import weapon

#Weapon_Tables Init
nowep = weapon.Weapon("None")
Pickaxe = weapon.Weapon("Pickaxe")
AR = weapon.Weapon("AR")
SCAR = weapon.Weapon("SCAR")
BURST = weapon.Weapon("BURST")
PUMP = weapon.Weapon("PUMP")
TAC = weapon.Weapon("TAC")
HEAVY = weapon.Weapon("HEAVY")
SMG = weapon.Weapon("SMG")
S_SMG = weapon.Weapon("S_SMG")
TOMMYGUN = weapon.Weapon("TOMMYGUN")
P90 = weapon.Weapon("P90")
HR = weapon.Weapon("HR")
BOLT = weapon.Weapon("BOLT")
SAR = weapon.Weapon("SAR")
ROCKET = weapon.Weapon("ROCKET")
NADES = weapon.Weapon("NADES")
HEAL = weapon.Weapon("HEAL")

#Settings;

#Debug:
Debug = True #Debug Mode, used for Output of Debug Info.
Debug_Movement = False #If Debug Mode is On, output Mouse Movement Info.
Debug_Buttons = True #If Debug Mode is On, output Mouse Button Activity.
Debug_Wheel = False #If Debug Mode is On, output Mouse Wheel Movement Info.
Debug_Keyboard = False #If Debug Mode is On, output Keyboard Activity.


#Features;

#General:
Option_AR_Perfect_Aim = False #If Enabled, causes constant First Shot Accuracy with AR when in ADS.
Option_Fast_Weapon_Switch = True #If Enabled, causes the Weapon Pullout Animation Cancel to be done on every weapon switch.
Option_Reload_Canceling = True #If Enabled, causes an Animation Cancel to occur if reloading when attempting to use.

#Quick Shoot:
Option_Quick_Shoot = True #If Enabled, allows an instant fire of a specific weapon choosen below.
if Option_Quick_Shoot:
    Quick_Shoot_Default_Weapon = PUMP
QuickShoot_Fire_Attempts = 40 #(Delay is 10ms so loop of 40 is around 400ms).

#Turbo Use:
Option_Turbo_Use = True #When Enabled, causes the use Key to Autofire when held down.
Delay_Turbo_Use = 0.060 #Time inbetween autofire + the natural 20ms delay for the keypress.

#Editing:
Option_Fast_Edit_Reset = True #When the Reset Edit Button is Pressed, Instantly exit edit mode.

#Ghost Peaking(TODO):
Use_Ghost_Peaking = True #If Enabled, uses a Modifier Key to enable fast crouching while shooting.
Ghost_Peak_Delay = 100 #Time between Ghost Peak Shots.

#Fast Farming(TODO):
Fast_Farming = False #If Enabled, Uses a Button to activate "Current Farming Exploit".
Turbofarm_Activation_Window = 300 #Delay from pressing "Farming_Button" to start Normal Farming instead of turbo.
Turbofarm_Fast_Swing_Delay = 200
Turbofarm_Emote_Time = 70
Turbofarm_Slow_Swing_Delay = 400
Turbofarm_Emote_Cancel_Delay = 300
STWAF = False #If Enabled, Switch to a Weapon of Choice after Farming.
STWAF_WeaponSlot = 1 #Weapon to Switch to after Farming.

#Anti Recoil:
Use_No_Recoil_ADS = True #If Enabled, Reduces Recoil when ADS Firing.
Use_No_Recoil_Hipfire = True #If Enabled, Reduces Recoil when Hip Firing.
Use_Jitter = True #If Enabled, Moves the Aim back and forth while firing.

#Weapon Slot Options
Default_Ingame_Weapon_Slot1 = AR
Default_Ingame_Weapon_Slot2 = PUMP
Default_Ingame_Weapon_Slot3 = SMG
Default_Ingame_Weapon_Slot4 = HR
Default_Ingame_Weapon_Slot5 = HEAL

#-------------------------------
#Keybinds;
#-------------------------------

#Macro Related;
#-------------------------------

#General:
Keybind_Fire = "left" #Key to Shoot Weapons.

#Key to Fast Fire Pre Set Weapon.
if Option_Quick_Shoot:
    Keybind_QuickShoot_Button = "x2"
    Keybind_QuickShoot_Button_Code = 0x06

Keybind_Enable = "middle" #Key to Enable/Disable the Macro.
Keybind_Edit = "alt" #Desired Key to go into edit mode.(Must have In-game set to alternate).
Keybind_Use = "f" #Desired key for Use.
if Option_Turbo_Use:
    Keybind_Use_Key_Code = 0x46

#Weapon Keybinds Part 1 (Macro)
#-----------------------------------------------
#Set these to the same as in-game if not using fast weapon switching.
#If using fast weapon switching set these to your desired keys for each slot-
#and scroll down to set the alternate in game key binds so the fast switch can work.
Keybind_Weapon_Slot1 = "2"
Keybind_Weapon_Slot2 = "3"
Keybind_Weapon_Slot3 = "4"
Keybind_Weapon_Slot4 = "z"
Keybind_Weapon_Slot5 = "x"
Keybind_Weapon_Pickaxe = "l"

#Ingame;
#These Keybinds should match the ones set in game.
#-------------------------------

#General:
Ingame_Keybind_Jump = "space"
Ingame_Keybind_Crouch = "shift" #Ingame Keybind set for Crouch.
Ingame_Keybind_Fire = "k" #Ingame Keybind set to Fire(Not Mouse1!!!).
if Option_Turbo_Use:
    Ingame_Keybind_Use = "i"
else:
    Ingame_Keybind_Use = Keybind_Use
Ingame_Keybind_Reload = "r"

#Weapon Keybinds Part 2 (In-game)
# Fast Weapon Switching:
#-------------------------------------------------
#Used ONLY when Option_Fast_Weapon_Switch is Enabled.
#Fast Weapon Switching is Required to have intermedary control over the weapon keys so-
#they need to be set to something different in-game.
#Ignore this Section if not using Fast Weapon Switching.
if Option_Fast_Weapon_Switch == True:
    Ingame_Weapon_Slot1 = "p"
    Ingame_Weapon_Slot2 = "6"
    Ingame_Weapon_Slot3 = "7"
    Ingame_Weapon_Slot4 = "8"
    Ingame_Weapon_Slot5 = "9"
    Ingame_Weapon_Pickaxe = "0"
else:
    Ingame_Weapon_Slot1 = Keybind_Weapon_Slot1
    Ingame_Weapon_Slot2 = Keybind_Weapon_Slot2
    Ingame_Weapon_Slot3 = Keybind_Weapon_Slot3
    Ingame_Weapon_Slot4 = Keybind_Weapon_Slot4
    Ingame_Weapon_Slot5 = Keybind_Weapon_Slot5
    Ingame_Weapon_Pickaxe = Keybind_Weapon_Pickaxe

#Other Weapons
Weapon_Trap = "5" #Actual In-game Keybind used to select traps.

#Building:
Ingame_Keybind_Ramp = "e" #Actual Ingame Keybind set for Ramp.
Ingame_Keybind_Wall = "x" #Actual Ingame Keybind set for Wall(Thumb Button 2 on Logetech G502).
Ingame_Keybind_Platform = "q" #Actual Ingame Keybind set for Platform.
Ingame_Keybind_Roof = "[" #Actual Ingame Keybind set for Roof.

#Editing:
Ingame_Keybind_Edit = "]" #Alternate Ingame Keybind set for Edit.
Ingame_Keybind_Edit_Reset = "right" #Actual Ingame Keybind set for Edit Reset.

#Emoting:
Keybind_EmoteWheel_Up = "up"
Keybind_EmoteWheel_Down = "down"
Keybind_Emote = "n"
Keybind_FastEmote1 = "f1"
Keybind_FastEmote2 = "f2"
Keybind_FastEmote3 = "f3"


#Globals:
global Running
global Enabled
global Current_Weapon
global Current_Loadout
global JitterFlop
global Editing
global Building
global Crouching
global Jumping
global Reloading
global Using

#Main Functions;

#Main Function: Gets run at start after init.
def main():
    # Print out a List of the Globals to Check Inits.
    DebugPrint('Printing Globals List:')
    DebugPrintGlobals()

    #Hook Logical Mouse Events.
    DebugPrint('Hooking Mouse Input')
    mouse.hook(Mouse_Input)

    #Hook Logical Keyboard Events.
    DebugPrint('Hooking Keyboard Input')
    keyboard.hook(Keyboard_Input)


#Helper Functions;

#Move Mouse:  Moves Mouse Relative to Current Location.
def Move_Mouse(x, y):
    windll.user32.mouse_event(
        c_uint(0x0001),
        c_uint(x),
        c_uint(y),
        c_uint(0),
        c_uint(0)
    )

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

def queryMousePosition():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return pt

#Cancel Anim: Cancels Weapon Switching Animation and Pickaxe Swings.
def Cancel_Anim(Weapon, Emote_Bind):
    #Emote to cancel pull out animation.
    keyboard.press(Emote_Bind);
    sleep(0.100);
    #Switch to Desired Weapon.
    Switch_to_Weapon(Weapon)
    sleep(0.020);
    keyboard.release(Emote_Bind);
    sleep(0.030);

#Converts Inventory Slot ID to actual weapon slot.
def InvSlot_to_WeaponKeybind(InvSlot):
    if InvSlot == 0:
        return Ingame_Weapon_Slot1
    if InvSlot == 1:
        return Ingame_Weapon_Slot2
    if InvSlot == 2:
        return Ingame_Weapon_Slot3
    if InvSlot == 3:
        return Ingame_Weapon_Slot4
    if InvSlot == 4:
        return Ingame_Weapon_Slot5

#Check Quick Shoot Button Function.
def isPressed_QuickShoot_Button():
    QuickShoot_Button_State = GetKeyState(Keybind_QuickShoot_Button_Code)
    DebugPrint("MB6:{0}".format(QuickShoot_Button_State))
    if QuickShoot_Button_State < 0:
        return True
    return False

def Toggle_Enabled():
    global Enabled
    if Enabled:
        Enabled = False
    else:
        Enabled = True
    DebugPrint("Enabled changed to {0}".format(Enabled))

#Debug PrintFunction: Print's Debug Output if Debug is Enabled.
def DebugPrint(s):
    if Debug:
        print("{0}:{1}".format(strftime("%H:%M:%S", gmtime()),s));

def DebugPrintGlobals():
    for k, v in globals().items():
        print("{0} = {1}".format(k, v))

#Turbo Use Function: Sends the Use Event as long as the Use key is held down.
def Turbo_Use():
    global Reloading
    global Using

    #Check if we are Reloading and want to use Reload Canceling.
    if Reloading and Option_Reload_Canceling:
        #Since we are canceling the Reload, set global to false.
        Reloading = False
        Using = False

        #Switch Directly to Pickaxe to Cancel the Reload.
        keyboard.send(Ingame_Weapon_Pickaxe);
        sleep(0.020);

        #Switch Back to the Weapon we had before the Animation Cancel.
        Switch_to_Weapon(Current_Weapon)

    #Check if we are'nt already Using.
    if not Using:
        Using = True
        DebugPrint("Detected Use Action")

        #Handle Turbo Use Action.
        if Option_Turbo_Use:

            overrun = 400
            #While Use Key is Down Loop.
            while True:
                #Overrun Protection.
                overrun = overrun - 1
                if overrun == 0:
                    break

                sleep(0.020) #Sleep To prevent GetKeyState Lockup.

                if windll.user32.GetKeyState(Keybind_Use_Key_Code) >= 0:
                    break

                DebugPrint("Sending Turbo Use Action")

                #Send Use Action.
                keyboard.press(Ingame_Keybind_Use)
                sleep(0.020)
                keyboard.release(Ingame_Keybind_Use)

                #Delay a Pre Set amount.
                sleep(Delay_Turbo_Use)
        else:
            print("Sending Use Action")
            keyboard.send(Ingame_Keybind_Use)
            sleep(0.080)


#Shooting Functions;

#Quick_Shoot: Quick Switches to Desired Weapon, pulls trigger and then switches back.
def Quick_Shoot(Weapon):
    global Editing
    global Building
    global Reloading

    #Set Building and Editing to False since we are now shooting.
    Building = False
    Editing = False

    #Store our Current Weapon so we can return to it after Quick.
    OldWeapon = Current_Weapon
    print("Insta Shooting {0}".format(Weapon.name))

    #Loop Until Button is Detected
    while True:
        if not isPressed_QuickShoot_Button():
            break #Break to Handle Quick Shoot.
        sleep(0.100)

    #Check if we are ADS
    MB2_State = GetKeyState(0x02)
    DebugPrint("Checking for MB2 or Reload")
    if MB2_State < 0 or Reloading:
        #We are ADS and trying to shoot with No Recoil.
        DebugPrint("Detected MB2 for ADS or Reloading")
        keyboard.send(Ingame_Weapon_Pickaxe);
        sleep(0.020);

    #Since we are shooting and have canceled any reload anim by now set it to false.
    Reloading = False

    #Emote to cancel pull out animation.
    Cancel_Anim(Weapon, Keybind_FastEmote3)

    count = 0
    #Loop while Attempting to Fire at a specified count.
    while True:
        count = count + 1
        if count == QuickShoot_Fire_Attempts:
            break #Break when we have hit the end of the delay.
        #Send Fire Event.
        keyboard.send(Ingame_Keybind_Fire)
        DebugPrint("Firing Fast")
        sleep(0.010)
    #Switch back to Weapon that was Active before the Fast Shoot.
    Switch_to_Weapon(OldWeapon)
    sleep(0.100)
    #Check if MB1 is still down.
    Handle_MB1_Down()

#Fire: Handles No Recoil and Anti Bloom
def Fire(ADS, Recoil):
    global JitterFlop
    global Crouching
    global Reloading

    #Init Locals
    local_Jitter = 0
    local_Jitter_ADS = 0

    #We are Shooting and not reloading so set global to False
    Reloading = False

    Local_Mouse_Pos = queryMousePosition()

    DebugPrint("Sending Simulated Fire Event")
    keyboard.press(Ingame_Keybind_Fire) #Send the input to fire the weapon
    sleep(0.020) #Delay to ensure the fire button gets recognized by the game

    if Current_Weapon.name != "Pickaxe" and (Current_Weapon.type == "AR" or Current_Weapon.type == "SMG"):

        #We are Shooting with Fire Button Down and without a Pickaxe
        if Current_Weapon.ControlFireRate:
            #Release the Fire Button if we are controlling Fire Rate
            keyboard.release(Ingame_Keybind_Fire)

        #If Fire has beeen called with the Recoil Bool then we can compensate
        if Recoil:
            #We are not Pickaxing and wanting to use Jitter so set it based on weapon
            if Use_Jitter:
                if JitterFlop == False:
                    JitterFlop = True
                    local_Jitter = Current_Weapon.jitter
                    local_Jitter_ADS = Current_Weapon.jitter_ADS
                else:
                    JitterFlop = False
                    local_Jitter = Current_Weapon.jitter * -1
                    local_Jitter_ADS = Current_Weapon.jitter_ADS * -1
            if Current_Weapon.firstshot == 0:
                Current_Weapon.firstshot = time()
                print("firstshot: {0}".format(time()))
                if ADS:
                    #mouse.move(local_Jitter_ADS, Current_Weapon.recoil_ADS_Static, absolute=False, duration=Current_Weapon.update_time_ADS)
                    Move_Mouse(local_Jitter_ADS, Current_Weapon.recoil_ADS_Init)
                else:
                    print("current coursor pos: X:{0}, Y:{1}".format(Local_Mouse_Pos.x, Local_Mouse_Pos.y))
                    Move_Mouse(local_Jitter, Current_Weapon.recoil_Hip_Init)
            else:
                difference = time() - Current_Weapon.firstshot
                print("Not firstshot, Difference: {0}".format(difference))
                if ADS:
                    #mouse.move(local_Jitter_ADS, Current_Weapon.recoil_ADS_Static, absolute=False, duration=Current_Weapon.update_time_ADS)
                    if difference >= Current_Weapon.recoil_ADS_Init_Delay:
                        Move_Mouse(local_Jitter_ADS, Current_Weapon.recoil_ADS_Static)
                    else:
                        Move_Mouse(local_Jitter_ADS, Current_Weapon.recoil_ADS_Init)
                else:
                    if difference >= Current_Weapon.recoil_Hip_Init_Delay:
                        print("current coursor pos: X:{0}, Y:{1}".format(Local_Mouse_Pos.x, Local_Mouse_Pos.y))
                        Move_Mouse(local_Jitter, Current_Weapon.recoil_Hip_Static)
                    else:
                        print("current coursor pos: X:{0}, Y:{1}".format(Local_Mouse_Pos.x, Local_Mouse_Pos.y))
                        Move_Mouse(local_Jitter, Current_Weapon.recoil_Hip_Init)
                #mouse.move(local_Jitter, Current_Weapon.recoil_Hip_Static, absolute=False, duration=Current_Weapon.update_time)
    DebugPrint("Done Sending Simulated Fire Event")

#Switch to Weapon: Finds Weapon in Current Loadout and Switches to it
def Switch_to_Weapon(Weapon):
    global Editing
    global Building
    global Current_Weapon
    global Reloading

    if Weapon == Current_Weapon and not Building and not Editing:
        return 2 #Return Error:2 Which means Weapon is Already Out.

    #Check if we are trying to switch to the Pickaxe.
    if Weapon.name == "Pickaxe":
        DebugPrint("Attempting to Switch Weapon to Pickaxe")

        #Send Event to Switch to Pickaxe.
        keyboard.send(Ingame_Weapon_Pickaxe)

        #Set the Global to the new weapon.
        Current_Weapon = Weapon

        #Reset Globals.
        Building = False
        Editing = False
        Reloading = False

        return 1 #Return 1 for Success.
    else: #We are trying to switch to another weapon other than the pickaxe.

        DebugPrint("Attempting to Switch Weapon\nSearching for Weapon:{0} in Current Loadout".format(Weapon.name))

        #Parse the Players Current Loadout for the Weapon.
        for i in range(len(Current_Loadout)):
            DebugPrint("Checking Inventory Slot{0}\nFound: {1}".format(i, Current_Loadout[i].name))

            if Current_Loadout[i].name == Weapon.name:
                DebugPrint("Found Weapon, Attempting Switch")

                #Switch to the Found weapon
                keyboard.send(InvSlot_to_WeaponKeybind(i))

                #Set the Global to the new weapon.
                Current_Weapon = Weapon

                #Reset Globals.
                Building = False
                Editing = False
                Reloading = False

                return 1 #Return 1 for Success.
            elif i >= 5: #We have reached the end of the Loadout.
                break #Break to return 3 Error.
    return 3 #Return 3 Which means Unknown Error.

#Handle Mouse Button 1 Down, Pretty much the Bread and Butter
#TODO need to find a better way to check MB1, MB2 and MB6 so it doesnt use Win32API
def Handle_MB1_Down():

    #Locals
    ADS = False
    overrun = 0

    #Debug Out
    DebugPrint("MB1 Event Detected")

    #Loop While MB1 is down and fast shoot button is not pressed
    while True:
        #Prevent Infinite Loop from an Overrun Error.
        overrun = overrun + 1
        if overrun >= 4000:
            Current_Weapon.firstshot = 0 #Reset First Shot Timer
            keyboard.release(Ingame_Keybind_Fire) #Relase the Actual Fire Key
            break #Break if we hit overrun

        #Use Win32API to get state of MB1 and MB2 because mouse.is_pressed() is retarted
        MB1_State = GetKeyState(0x01)
        DebugPrint("Checking if MB1 is Still Down")
        #Check if MB1 is still down both Logically and Hardware Based
        if MB1_State >= 0 and GetAsyncKeyState(0x01) == 0:
            Current_Weapon.firstshot = 0 #Reset First Shot Timer
            DebugPrint("MB1 Button Released, State: {0}".format(MB1_State))
            keyboard.release(Ingame_Keybind_Fire) #Relase the Actual Fire Key
            break #Break if we are not pressing MB1 Anymore
        else:
            DebugPrint("MB1 Still Detected, State: {0}".format(MB1_State))

        #Check if we are in Pickaxe Mode
        if Current_Weapon.name == "Pickaxe":

            #Check if Quick Shoot Button is Pressed
            if isPressed_QuickShoot_Button() and Option_Quick_Shoot:
                break #Break to Handle Quick Shoot

            DebugPrint("Swinging Pickaxe")
            Fire(False, False) # Send Command to Swing Pickaxe

            #FastFarming would go here

            #if we are using Quick shoot then we need to check every 10ms during the pickaxe swing delay
            if Option_Quick_Shoot:
                count = 20 # 20x10ms = 200ms (Current Optimal Pickaxe Swingtime)
                while True:
                    count = count - 1
                    if count <= 0:
                        break #Break to Swing the Axe again
                    if isPressed_QuickShoot_Button():
                        break #Break to Handle Quick Shoot
                    sleep(0.010)
            else: #Not using Quickshoot so just delay 200ms
                sleep(0.200)

        else: # Trying to Fire something that is not a pickaxe

            #Check if we are trying to Quick Shoot
            if isPressed_QuickShoot_Button() and Option_Quick_Shoot:
                Current_Weapon.firstshot = 0 #Reset First Shot Timer
                if not Current_Weapon.ControlFireRate: #Fire Key may still be Down from Last Fire() so release it.
                    keyboard.release(Ingame_Keybind_Fire)
                break #Break to Handle Quick Shoot

            #Check if we are ADS.
            MB2_State = GetKeyState(0x02)
            DebugPrint("Checking for MB2")
            #Check for MB2 both Logically and Hardware Based
            if MB2_State < 0 and GetAsyncKeyState(0x02) == 1:
                #We are ADS and trying to shoot with No Recoil
                DebugPrint("MB2 is Detected, Setting ADS to True")
                ADS = True
            else:
                ADS = False

            #Check if we should use anti recoil for hipfire
            if Use_No_Recoil_Hipfire and not ADS:
                #Hipfire No Recoil
                DebugPrint("Hipfire Reducing")
                Fire(False, True)

                #Check if we are trying to Quick Shoot
                if isPressed_QuickShoot_Button() and Option_Quick_Shoot:
                    Current_Weapon.firstshot = 0 #Reset First Shot Timer
                    if not Current_Weapon.ControlFireRate: #Fire Key may still be Down from Last Fire() so release it.
                        keyboard.release(Ingame_Keybind_Fire)
                    break #Break to Handle Quick Shoot

                #Delay for the set amount of time in the Weapon Def
                sleep(Current_Weapon.update_time)
            elif ADS and Use_No_Recoil_ADS:
                #ADS, No Recoil
                DebugPrint("ADS Reducing")
                Fire(True, True)

                #Check if we are trying to Quick Shoot
                if isPressed_QuickShoot_Button() and Option_Quick_Shoot:
                    Current_Weapon.firstshot = 0 #Reset First Shot Timer
                    if not Current_Weapon.ControlFireRate: #Fire Key may still be Down from Last Fire() so release it.
                        keyboard.release(Ingame_Keybind_Fire)
                    break #Break to Handle Quick Shoot

                #Delay for the set amount of time in the Weapon Def
                sleep(Current_Weapon.update_time_ADS)
            else:
                #No Recoil Mode Off
                DebugPrint("Not Reducing")
                Fire(False, False)

                #Check if we are trying to Quick Shoot
                if isPressed_QuickShoot_Button() and Option_Quick_Shoot:
                    Current_Weapon.firstshot = 0 #Reset First Shot Timer
                    if not Current_Weapon.ControlFireRate: #Fire Key may still be Down from Last Fire() so release it.
                        keyboard.release(Ingame_Keybind_Fire)
                    break #Break to Handle Quick Shoot

                #Delay for the set amount of time in the Weapon Def
                sleep(Current_Weapon.delay_time)

           # if Current_Weapon.ControlFireRate == False:
           #     Current_Weapon.firstshot = 0
           #     keyboard.release(Ingame_Keybind_Fire)
           # break

#Keyboard Hook Callback Function: Gets run when Keyboard activity is detected.
def Keyboard_Input(Keyboard_Event):
    global Editing
    global Building
    global Crouching
    global Jumping
    global Reloading
    global Using

    if type(Keyboard_Event) == keyboard._keyboard_event.KeyboardEvent:
        #Debug Output
        if Debug_Keyboard:
            DebugPrint("Keyboard_Event: Key:{0},Scancode:{1}, Action:{2},Modifier:{3} Time:{4}".format(Keyboard_Event.name,Keyboard_Event.scan_code, Keyboard_Event.event_type, Keyboard_Event.modifiers, Keyboard_Event.time))

        #Weapon Slot 1
        if Keyboard_Event.name == Keybind_Weapon_Slot1 and Keyboard_Event.event_type == "down" and Enabled:

            #Get Weapon in Slot 1 of Inventory
            TargetWeapon = Current_Loadout[0]

            #Check if we should try and Fast Switch the Weapon
            if Option_Fast_Weapon_Switch and TargetWeapon.type != "Heal" and TargetWeapon.type != "Nade":
                if TargetWeapon != Current_Weapon or (Building or Editing):
                    Cancel_Anim(TargetWeapon, Keybind_FastEmote3)
                else:
                    DebugPrint("Error when attempting to fast switch: Weapon Already Selected")
            else:
                #If we are in the middle of a reload then Cancel by switching to Pickaxe
                if Reloading:
                    Reloading = False
                    keyboard.send(Ingame_Weapon_Pickaxe);
                    sleep(0.020);
                Switch_to_Weapon(TargetWeapon)

        #Weapon Slot 2
        elif Keyboard_Event.name == Keybind_Weapon_Slot2 and Keyboard_Event.event_type == "down" and Enabled:

            #Get Weapon in Slot 2 of Inventory
            TargetWeapon = Current_Loadout[1]

            #Check if we should try and Fast Switch the Weapon
            if Option_Fast_Weapon_Switch and TargetWeapon.type != "Heal" and TargetWeapon.type != "Nade":
                if TargetWeapon != Current_Weapon or (Building or Editing):
                    Cancel_Anim(TargetWeapon, Keybind_FastEmote3)
                else:
                    DebugPrint("Error when attempting to fast switch: Weapon Already Selected")
            else:
                #If we are in the middle of a reload then Cancel by switching to Pickaxe
                if Reloading:
                    Reloading = False
                    keyboard.send(Ingame_Weapon_Pickaxe);
                    sleep(0.020);
                Switch_to_Weapon(TargetWeapon)

        #Weapon Slot 3
        elif Keyboard_Event.name == Keybind_Weapon_Slot3 and Keyboard_Event.event_type == "down" and Enabled:

            #Get Weapon in Slot 3 of Inventory
            TargetWeapon = Current_Loadout[2]

            #Check if we should try and Fast Switch the Weapon
            if Option_Fast_Weapon_Switch and TargetWeapon.type != "Heal" and TargetWeapon.type != "Nade":
                if TargetWeapon != Current_Weapon or (Building or Editing):
                    Cancel_Anim(TargetWeapon, Keybind_FastEmote3)
                else:
                    DebugPrint("Error when attempting to fast switch: Weapon Already Selected")
            else:
                #If we are in the middle of a reload then Cancel by switching to Pickaxe
                if Reloading:
                    Reloading = False
                    keyboard.send(Ingame_Weapon_Pickaxe);
                    sleep(0.020);
                Switch_to_Weapon(TargetWeapon)

        #Weapon Slot 4
        elif Keyboard_Event.name == Keybind_Weapon_Slot4 and Keyboard_Event.event_type == "down" and Enabled:

            #Get Weapon in Slot 4 of Inventory
            TargetWeapon = Current_Loadout[3]

            #Check if we should try and Fast Switch the Weapon
            if Option_Fast_Weapon_Switch and TargetWeapon.type != "Heal" and TargetWeapon.type != "Nade":
                if TargetWeapon != Current_Weapon or (Building or Editing):
                    Cancel_Anim(TargetWeapon, Keybind_FastEmote3)
                else:
                    DebugPrint("Error when attempting to fast switch: Weapon Already Selected")
            else:
                #If we are in the middle of a reload then Cancel by switching to Pickaxe
                if Reloading:
                    Reloading = False
                    keyboard.send(Ingame_Weapon_Pickaxe);
                    sleep(0.020);
                Switch_to_Weapon(TargetWeapon)

        #Weapon Slot 5
        elif Keyboard_Event.name == Keybind_Weapon_Slot5 and Keyboard_Event.event_type == "down" and Enabled:

            #Get Weapon in Slot one of Inventory
            TargetWeapon = Current_Loadout[4]

            #Check if we should try and Fast Switch the Weapon
            if Option_Fast_Weapon_Switch and TargetWeapon.type != "Heal" and TargetWeapon.type != "Nade":
                if TargetWeapon != Current_Weapon or (Building or Editing):
                    Cancel_Anim(TargetWeapon, Keybind_FastEmote3)
                else:
                    DebugPrint("Error when attempting to fast switch: Weapon Already Selected")
            else:
                #If we are in the middle of a reload then Cancel by switching to Pickaxe
                if Reloading:
                    Reloading = False
                    keyboard.send(Ingame_Weapon_Pickaxe);
                    sleep(0.020);
                Switch_to_Weapon(TargetWeapon)

        #Weapon Slot Pickaxe
        elif Keyboard_Event.name == Keybind_Weapon_Pickaxe and Keyboard_Event.event_type == "down" and Enabled:

            #Check if we should try and Fast Switch the Weapon
            if Option_Fast_Weapon_Switch:
                if Pickaxe != Current_Weapon or (Building or Editing):
                    Cancel_Anim(Pickaxe, Keybind_FastEmote3)
                else:
                    DebugPrint("Error when attempting to fast switch: Weapon Already Selected")
            else:
                Switch_to_Weapon(Pickaxe)

        #Edit Keybind Handling
        elif Keyboard_Event.name == Keybind_Edit and Keyboard_Event.event_type == "down" and Enabled:
            #Set the Edit flag
            if not Editing:
                keyboard.send(Ingame_Keybind_Edit)
            DebugPrint("Detected Editing")
            Editing = True
        elif Keyboard_Event.name == Keybind_Edit and Keyboard_Event.event_type == "up" and Enabled:
            #Set the Edit flag
            if Editing:
                keyboard.send(Ingame_Keybind_Edit)
            DebugPrint("Finished Editing")
            Editing = False

        #Building Keybind Handling
        elif Keyboard_Event.name == Ingame_Keybind_Ramp and Keyboard_Event.event_type == "down" and Enabled:
            DebugPrint("Detected Building")
            #Set the Building flag
            Building = True
        elif Keyboard_Event.name == Ingame_Keybind_Platform and Keyboard_Event.event_type == "down" and Enabled:
            DebugPrint("Detected Building")
            #Set the Building flag
            Building = True
        elif Keyboard_Event.name == Ingame_Keybind_Roof and Keyboard_Event.event_type == "down" and Enabled:
            DebugPrint("Detected Building")
            #Set the Building flag
            Building = True

        #Cruching Keybind Handling
        elif Keyboard_Event.name == Ingame_Keybind_Crouch and Keyboard_Event.event_type == "down" and Enabled:
            DebugPrint("Detected Crouching")
            #Set the Building flag
            if Crouching:
                Crouching = False
            else:
                Crouching = True

        #Jumping Keybind Handling
        elif Keyboard_Event.name == Ingame_Keybind_Jump and Keyboard_Event.event_type == "down" and Enabled:
            DebugPrint("Detected Jumping")
            #Set the Jumping Global to Current Time.
            Jumping = time()
            #Set the Crouching Global to False since if jumping we are not crouching.
            Crouching = False

        #Reload Keybind Handling
        elif Keyboard_Event.name == Ingame_Keybind_Reload and Keyboard_Event.event_type == "down" and Enabled:
            DebugPrint("Detected Reloading")
            Reloading = True

        #Use Keybind Handling
        elif Keyboard_Event.name == Keybind_Use and Keyboard_Event.event_type == "down" and Enabled:
            Turbo_Use()
        elif Keyboard_Event.name == Keybind_Use and Keyboard_Event.event_type == "up" and Enabled:
            Using = False
            DebugPrint("Detected End of Use Action")

#Mouse Hook Callback Function: Gets run when Mouse activity is detected.
def Mouse_Input(Mouse_Event):
    global Editing
    global Building

    if Running:
        #Mouse Wheel Event.
        if type(Mouse_Event) == mouse._mouse_event.MoveEvent:
            if Debug_Movement:
                DebugPrint("Move_Event: x:{0}, y:{1}, Time:{2}".format(Mouse_Event.x, Mouse_Event.y, Mouse_Event.time))

        #Mouse Button Event.
        elif type(Mouse_Event) == mouse._mouse_event.ButtonEvent:

            if Debug_Buttons:
                DebugPrint("Button_Event: Type:{0}, Button:{1}, Time:{2}".format(Mouse_Event.event_type, Mouse_Event.button, Mouse_Event.time))

            #Handle MouseButton1 Down Event
            if Mouse_Event.event_type == "down" and Mouse_Event.button == Keybind_Fire and Enabled:
                #Only Handle MB1 for Weapons so if building or editing then we dont want to control it.
                if not Editing and not Building:
                    Handle_MB1_Down()

            #Handle MouseButton2 Down Event
            elif Mouse_Event.event_type == "down" and Mouse_Event.button == Ingame_Keybind_Edit_Reset and Enabled:
                #Fast Edit Reset:
                if Editing and Option_Fast_Edit_Reset:

                    #If we are editing and right click with Fast Reset enabled then exit edit mode.
                    keyboard.send(Ingame_Keybind_Edit)
                    DebugPrint("Finished Editing with Quick Reset")
                    Editing = False

            #Handle Quick Switch Weapon
            elif Mouse_Event.event_type == "down" and Mouse_Event.button == Keybind_QuickShoot_Button and Enabled:
                Quick_Shoot(PUMP)

           #Handle Enable Eventv
            elif Mouse_Event.event_type == "down" and Mouse_Event.button == Keybind_Enable:
                Toggle_Enabled()

            #Handle Building Event
            elif Mouse_Event.event_type == "down" and Mouse_Event.button == Ingame_Keybind_Wall:
                DebugPrint("Detected Building")
                Building = True

        elif type(Mouse_Event) == mouse._mouse_event.WheelEvent:
            if Debug_Wheel:
                DebugPrint("Wheel_Event: Delta:{0}, Time:{1}".format(Mouse_Event.delta, Mouse_Event.time))

def Init_Weapons():
    #AR
    #Info;
    #Advertised Fire Rate = 181.81ms (5.5)
    AR.recoil_Hip_Init = 21
    AR.recoil_Hip_Init_Delay = 0.340
    AR.recoil_Hip_Static = 23
    AR.recoil_ADS_Init = 30
    AR.recoil_ADS_Init_Delay = 0.330
    AR.recoil_ADS_Static = 1
    AR.jitter = 0
    AR.jitter_ADS = 0
    AR.update_time = 0.040 #+0.020
    if Option_AR_Perfect_Aim:
        AR.update_time_ADS = 0.330
    else:
        AR.update_time_ADS = 0.040
    AR.firstshot = 0.0
    AR.ControlFireRate = True
    AR.type = "AR"
    #SCAR
    SCAR.recoil_Hip_Init = 21
    SCAR.recoil_Hip_Init_Delay = 0.340
    SCAR.recoil_Hip_Static = 23
    SCAR.recoil_ADS_Init = 30
    SCAR.recoil_ADS_Init_Delay = 0.330
    SCAR.recoil_ADS_Static = 1
    SCAR.jitter = 0
    SCAR.jitter_ADS = 0
    SCAR.update_time = 0.040
    SCAR.update_time_ADS = 0.330
    SCAR.firstshot = 0.0
    SCAR.ControlFireRate = True
    SCAR.type = "AR"
    #BURST
    BURST.recoil_Hip_Init = 150
    BURST.recoil_Hip_Static = 100
    BURST.recoil_ADS_Init = 125
    BURST.recoil_ADS_Static = 70
    BURST.jitter = 2
    BURST.jitter_ADS = 2
    BURST.update_time = 10
    BURST.update_time_ADS = 10
    BURST.firstshot = 0
    BURST.type = "AR"
    #PUMP
    PUMP.firstshot = 0
    PUMP.type = "Shotgun"
    #TAC
    TAC.firstshot = 0
    TAC.type = "Shotgun"
    #HEAVY
    HEAVY.firstshot = 0
    HEAVY.type = "Shotgun"
    #SMG
    SMG.recoil_Hip_Init = 4
    SMG.recoil_Hip_Init_Delay = 0.160
    SMG.recoil_Hip_Static = 8
    SMG.recoil_ADS_Init = 5
    SMG.recoil_ADS_Init_Delay = 0.160
    SMG.recoil_ADS_Static = 1
    SMG.jitter = 10
    SMG.jitter_ADS = 2
    SMG.update_time = 0.010
    SMG.update_time_ADS = 0.010
    SMG.firstshot = 0.0
    SMG.ControlFireRate = False
    SMG.type = "SMG"
    #SILENCEDSMG
    S_SMG.recoil_Hip_Init = 150
    S_SMG.recoil_Hip_Static = 100
    S_SMG.recoil_ADS_Init = 125
    S_SMG.recoil_ADS_Static = 70
    S_SMG.jitter = 2
    S_SMG.jitter_ADS = 2
    S_SMG.update_time = 10
    S_SMG.update_time_ADS = 10
    S_SMG.firstshot = 0
    S_SMG.type = "SMG"
    #TOMMYGUN
    TOMMYGUN.recoil_Hip_Init = 150
    TOMMYGUN.recoil_Hip_Static = 100
    TOMMYGUN.recoil_ADS_Init = 125
    TOMMYGUN.recoil_ADS_Static = 70
    TOMMYGUN.jitter = 2
    TOMMYGUN.jitter_ADS = 2
    TOMMYGUN.update_time = 10
    TOMMYGUN.update_time_ADS = 10
    TOMMYGUN.firstshot = 0
    TOMMYGUN.type = "SMG"
    #P90
    P90.recoil_Hip_Init = 150
    P90.recoil_Hip_Static = 100
    P90.recoil_ADS_Init = 125
    P90.recoil_ADS_Static = 70
    P90.jitter = 2
    P90.jitter_ADS = 2
    P90.update_time = 10
    P90.update_time_ADS = 10
    P90.firstshot = 0
    P90.type = "SMG"
    #Hunting Rifle
    HR.firstshot = 0
    HR.type = "Sniper"
    #Bolt Rifle
    BOLT.firstshot = 0
    BOLT.type = "Sniper"
    #Scoped AR
    SAR.firstshot = 0
    SAR.type = "Sniper"
    #Rockets
    ROCKET.firstshot = 0
    ROCKET.type = "Splode"
    #Grenades
    NADES.firstshot = 0
    NADES.type = "Nade"
    #Healing
    HEAL.type = "Heal"

#Detect if we are being run
if __name__ == '__main__':
    DebugPrint('Starting FortBolt\n')
    DebugPrint('Initalizing Globals\n')
    Running = True
    Enabled = False
    JitterFlop = False
    Editing = False
    Building = False
    Crouching = False
    Reloading = False
    Using = False
    Jumping = 0.0
    Init_Weapons()
    Current_Weapon = Pickaxe
    Current_Loadout = [AR, PUMP, SMG, HR, HEAL]

    try:
        main()
        exit
    except ValueError:
        print("Error: Could not Run Script")