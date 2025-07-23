### Turn-based game loop:

The game operates in short, repeatable rounds:
1. Countdown 
2. Player performs a hand gesture (rock/paper/scissors)
3. Gesture is recognized via camera input
4. Result shown: Win / Lose / Draw + gesture accuracy + score + feedback visuals

***Each round lasts ~10–15 seconds to maintain user focus and reduce fatigue.***
***Each game consists of multiple rounds. In each round, players may gain or lose points. A player wins the game upon reaching a score of 15; otherwise, the game continues until the required score is reached.***

🧠 _**Psychology Principle**_: 
	Short feedback loops increase motivation through **frequent, clear reinforcement**.
	Visual and audio effect drawing attention and immersive

### Computer opponent & Strategic variants

#### ***Player competes against computer with dynamic difficulty:***

1. - When the player has **low score or is losing**, the opponent gestures are generated **fully randomly**, making it easier to get score.
2. As the player’s score **increases** or approaches the **match-winning threshold**, the system introduces a:
    > `chance = current_score * 5%`   probability to **force a player loss**, increasing challenge under tension.

3. If a player is consistently **winning**, forced-loss probability increases with **overall win rate**.
4. Conversely, if the player enters **negative score** (from consecutive losses), the opponent is more likely to throw a gesture that ensures a **player win**.

🧠 **_Psychology support**_: 
	Maintains challenge while preventing frustration or boredom—aligned with the “Flow Theory” and player self-efficacy.

#### ***PVP mode using matching machoism to mach opponent at a comparable level***
- A planned multiplayer (PVP) mode will use **matchmaking algorithms** to pair patients with others of **comparable skill level** or **rehabilitation stage**.
- This prevents discouraging matchups and supports **social motivation**.

🧠 **_HCI & Psychology_**:
	Leveraging **social presence** and **competitive but fair** challenges increases engagement and emotional investment.

### Selective gamemode

Patients can choose their daily training mode as either:
 **Time-based**: The training ends after playing for a fixed duration  
 **or**
 **Game-based**: The training ends after winning a fixed number of games
			 *The gesture accuracy have higher score weight in this mode*

**Note**: Patients should not worry about endless gameplay in game-based mode—dynamic difficulty ensures the game always converges toward match completion.

🧠 _**Psychology**_: 
	Choice-based modes give users **autonomy**, supporting intrinsic motivation and better adherence.

### Motion Quality Scoring

- Each gesture is evaluated beyond correctness:
    
    - **ROM estimation** (how well the hand opens/closes)
        
    - **Reaction time** (measured from countdown to gesture)
        
    - **Consistency** of form and stability
        
- Players receive real-time ratings: Good / Great / Perfect with color-coded progress bars.
    

> 🧠 _HCI + Psych Principle_: Concrete feedback builds **self-efficacy** and reinforces proper motion

### **Multimodal Feedback System**

- Audio: Sounds for success/failure, countdown cues, victory jingles
    
- Visual: Gesture box changes color, icons glow or shake for emphasis
    
- Text: Encouraging words like “Well Done!”, “Great Move!”
    

🧠 **HCI Principle**: 
	Combining **audio + visual + textual feedback** reduces cognitive load and improves clarity.
	**Visualization of progress** is key for long-term engagement and motivation.