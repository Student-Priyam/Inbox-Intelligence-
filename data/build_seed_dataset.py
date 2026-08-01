"""
build_seed_dataset.py
----------------------
Generates data/training_data.csv — a labeled seed dataset used by
train_model.py to fine-tune the email classifier.

This is a STARTING POINT, not a finished dataset. It gives the trainer
enough labeled examples per category to produce a working model out of
the box. For real accuracy gains over the old zero-shot approach, you
should append real (anonymized) examples from your own inbox to
training_data.csv over time — the more varied, real-world phrasing the
model sees, the better it generalizes.

Run:
    python data/build_seed_dataset.py
"""

import csv
import itertools
import random
from pathlib import Path

random.seed(42)

OUT_PATH = Path(__file__).parent / "training_data.csv"

# Each category maps to a list of (subject, body) template pairs.
# Written to sound like real student-inbox emails, deliberately varied
# in phrasing so the model learns the *concept* per category rather than
# memorizing the same keywords the rule-based layer already catches.

SAMPLES = {
    "Urgent": [
        ("Action required: confirm your seat for tomorrow's viva",
         "Please confirm your attendance for tomorrow's viva voce by 6 PM today. "
         "Seats not confirmed will be reassigned and you will not be able to appear."),
        ("Interview scheduled for Thursday, 10 AM",
         "Your interview with the hiring panel has been scheduled for Thursday at "
         "10 AM. Please reply to confirm your availability by end of day."),
        ("Your visa document is missing — submit by Friday",
         "We noticed your visa supporting document is still missing from your "
         "application. Please upload it by Friday or your application will be "
         "withdrawn."),
        ("Reschedule notice: Exam moved to 9 AM sharp",
         "Your semester exam has been moved from 2 PM to 9 AM sharp tomorrow. "
         "Please make arrangements to reach the exam hall on time."),
        ("Fee payment overdue — account will be blocked",
         "Your semester fee payment is overdue. Please clear the dues within 24 "
         "hours to avoid your student account being blocked."),
        ("Please sign and return the internship offer today",
         "We need your signed offer letter returned to HR by 5 PM today to lock "
         "in your internship start date."),
        ("Your flight check-in closes in 3 hours",
         "This is a reminder that online check-in for your flight tomorrow "
         "closes in 3 hours. Please complete it to secure your seat."),
        ("Library book overdue — fine accruing",
         "The book you borrowed is now overdue and a daily fine is accruing on "
         "your account. Please return it at the earliest to avoid a hold on your ID."),
        ("Immediate response needed: thesis committee change",
         "Your thesis committee has requested a change to your defense date. "
         "Please respond today so we can finalize the new schedule."),
        ("Password reset required before you can log in again",
         "Your student portal password has expired for security reasons. You "
         "must reset it before your next login or you will be locked out."),
        ("Final reminder: submit your project report by 5 PM",
         "This is the final reminder to submit your project report before the "
         "5 PM deadline today. Late submissions will not be accepted."),
        ("Your account will be deactivated in 24 hours",
         "Please verify your enrollment status within the next 24 hours or "
         "your student account access will be deactivated."),
        ("Emergency: campus closed tomorrow due to weather",
         "Due to severe weather conditions, campus will be closed tomorrow. "
         "All exams and classes are rescheduled — check the portal for new timings."),
        ("Time-sensitive: scholarship application closes tonight",
         "The scholarship application portal closes at midnight tonight. "
         "Submit all required documents before then to be considered."),
        ("Your seat is at risk — pay the balance today",
         "Your seat allotment will be cancelled if the remaining balance is "
         "not paid by the end of today."),
        ("Last call: RSVP for the placement orientation",
         "This is the last call to RSVP for tomorrow's placement orientation. "
         "Unconfirmed students will not be allowed entry."),
        ("Action needed: sign the updated hostel agreement",
         "The updated hostel agreement must be signed and returned to the "
         "warden's office by tomorrow morning."),
        ("Your appeal deadline is in 2 hours",
         "The window to submit an appeal for your grade closes in 2 hours. "
         "Submit any supporting documents before then."),
        ("Immediate action: your badge access has been revoked",
         "Your building access badge has been temporarily revoked pending "
         "verification. Visit the security office today to resolve this."),
        ("Confirm your presence for the surprise inspection",
         "An inspection team will visit your lab tomorrow morning. Please "
         "confirm your presence by replying within the hour."),
        ("Overdue: internal assessment marks not submitted",
         "Your internal assessment marks are overdue. Submit them before end "
         "of day to avoid a hold on result processing."),
        ("Reschedule alert: your flight has been moved up",
         "Your connecting flight has been moved up by two hours. Please "
         "reach the airport earlier than originally planned."),
        ("Critical: server migration happening tonight",
         "A critical server migration is scheduled tonight. Save all your "
         "work and log off before 11 PM to avoid data loss."),
        ("Please respond before we release your seat",
         "We have not received your confirmation. Respond within the next "
         "few hours or your seat will be released to the waitlist."),
        ("Your document verification appointment is today",
         "Your document verification appointment is scheduled for today at "
         "3 PM. Please bring the originals and arrive 15 minutes early."),
        ("Last chance to opt in for the hostel allotment",
         "This is your last chance to opt in for hostel allotment this "
         "semester. The form closes in a few hours."),
        ("Payment failed — retry now to avoid losing your slot",
         "Your payment for the workshop did not go through. Retry now, as "
         "slots are limited and filling quickly."),
        ("Your ID card must be collected by 4 PM today",
         "Your new ID card is ready for collection at the admin office. "
         "Please collect it before 4 PM today."),
        ("Urgent correction needed on your submitted form",
         "We found an error on your submitted form that needs to be "
         "corrected today to avoid delays in processing."),
        ("Power outage scheduled — back up your work now",
         "A scheduled power outage will affect the lab in the next hour. "
         "Please save and back up your work immediately."),
    ],
    "Job/Internship": [
        ("New internship alert: Software Engineer Intern at a fintech startup",
         "A new internship matching your profile was just posted. Software "
         "Engineer Intern, remote, 3-month duration. View the listing and apply."),
        ("Your application status has been updated",
         "Your application for the Data Analyst role has moved to the next "
         "stage. The recruiter will reach out to schedule a screening call."),
        ("We'd like to connect about an opportunity at our company",
         "Hi, I came across your profile and think you'd be a great fit for an "
         "opening on our engineering team. Would you be open to a quick chat?"),
        ("5 new jobs matching 'frontend developer' near you",
         "We found new openings that match your saved search. Companies "
         "hiring this week include three mid-size product firms."),
        ("Your resume was viewed by a recruiter",
         "A recruiter at a growing startup viewed your profile this week and "
         "may reach out about a relevant opening."),
        ("Campus placement drive: register by tonight",
         "The placement cell is organizing a campus drive next week. Students "
         "interested in participating should register through the portal."),
        ("Congratulations — you've been shortlisted for the next round",
         "You have been shortlisted for the technical interview round. Further "
         "details about the schedule will follow shortly."),
        ("Summer internship program applications now open",
         "Applications for our 10-week summer internship program are now "
         "open. Explore tracks in engineering, product, and design."),
        ("Thank you for applying — here's what happens next",
         "We've received your application and our team will review it within "
         "two weeks. You'll hear from us either way."),
        ("New referral opportunity from your network",
         "Someone in your network referred you for an open position. Check "
         "out the role details and let them know if you're interested."),
        ("You're invited to a virtual hiring event",
         "Join our virtual hiring event next week to meet recruiters from "
         "several companies hiring for entry-level roles."),
        ("Your profile matches 3 new internships",
         "Based on your profile, we found three new internship listings "
         "this week that match your interests."),
        ("Interview feedback is ready to view",
         "Feedback from your recent interview round is now available in "
         "your applicant dashboard."),
        ("Offer letter attached — please review",
         "Congratulations! Please find your offer letter attached. Review "
         "the terms and let us know if you have questions."),
        ("Recruiter wants to schedule a call with you",
         "A recruiter reviewed your application and would like to schedule "
         "a short call this week to discuss next steps."),
        ("New job alert: openings at companies you follow",
         "Companies you follow just posted new openings that match your "
         "saved preferences."),
        ("Your internship certificate is ready for download",
         "Your internship completion certificate has been generated and is "
         "available for download from the portal."),
        ("Apply now: walk-in interviews this Saturday",
         "Walk-in interviews are being held this Saturday for multiple "
         "roles. Bring your resume and a copy of your ID."),
        ("Update on your job application timeline",
         "Here is an update on where your application stands and what to "
         "expect over the next two weeks."),
        ("You've been added to the talent pool",
         "Your profile has been added to our talent pool for future "
         "openings that match your skill set."),
        ("Reminder: complete your candidate profile",
         "Complete your candidate profile to be considered for upcoming "
         "internship openings this season."),
        ("Congratulations on completing round 1!",
         "You've cleared the first round of interviews. Details for the "
         "next round will be shared shortly."),
        ("New career fair announced for next month",
         "A career fair featuring multiple recruiters will be held next "
         "month. Registration is now open."),
        ("Your stipend details for the internship",
         "Please find attached the stipend and joining details for your "
         "upcoming internship."),
        ("Take-home assignment for the next round",
         "As part of the next interview round, please complete the "
         "attached take-home assignment within 5 days."),
        ("Company X is hiring for summer internships",
         "Company X just opened applications for their summer internship "
         "program in your field of interest."),
        ("Your background verification has started",
         "Your background verification process for the offered role has "
         "started and may take a few business days."),
        ("Invitation to connect with our alumni recruiter",
         "One of our alumni now recruiting at a growing company would like "
         "to connect with you about opportunities."),
        ("Internship extended — action needed to continue",
         "Your internship extension has been approved. Please confirm your "
         "continued availability by replying to this email."),
        ("Your application was shared with the hiring manager",
         "Your application has been forwarded to the hiring manager for "
         "further review."),
    ],
    "Follow-Up": [
        ("Following up on our conversation last week",
         "Just checking in on the notes I shared last week — let me know if "
         "you had a chance to look them over and if you have any questions."),
        ("Could you review this draft when you get a chance?",
         "I've attached the updated draft. Whenever you have a moment this "
         "week, could you take a look and share your thoughts?"),
        ("Quick check-in: any update on the group project slides?",
         "Wanted to check if you've had time to finish your section of the "
         "slides. No rush, just want to plan our next meeting."),
        ("Reminder to send over the signed form",
         "Just a friendly reminder to send over the signed form whenever "
         "convenient this week so we can keep things moving."),
        ("Can you confirm receipt of the documents I sent?",
         "Wanted to make sure the documents came through okay. Let me know "
         "if you can confirm receipt in the next day or so."),
        ("Following up: are you still interested in the study group?",
         "We're forming a study group for the upcoming exam. Let me know by "
         "tomorrow if you'd like to join so I can add you to the chat."),
        ("Circling back on the feedback you promised",
         "Hey, circling back on this — did you get a chance to jot down "
         "feedback on my sample chapter? Whenever works for you."),
        ("Waiting on your approval to proceed",
         "The next step is ready to go as soon as you approve the plan. Let "
         "me know if you have questions or if it looks good to proceed."),
        ("Did my last email come through?",
         "Not sure if my earlier message landed in your inbox. Wanted to "
         "follow up in case it got buried — let me know your thoughts."),
        ("Please review and approve the meeting notes",
         "Sharing the notes from our sync earlier. Please review and approve "
         "them so I can circulate to the rest of the group."),
        ("Just checking back on this",
         "Wanted to check back on this whenever you get a moment — no rush "
         "at all."),
        ("Any thoughts on the proposal I sent?",
         "Curious if you've had a chance to look over the proposal I sent "
         "last week."),
        ("Following up on my earlier question",
         "Following up on the question I asked earlier — happy to explain "
         "further if helpful."),
        ("Still waiting to hear back on this",
         "Just following up since I haven't heard back yet. Let me know if "
         "you need anything else from my end."),
        ("Checking in before our meeting next week",
         "Wanted to check in ahead of our meeting next week to see if "
         "there's anything you'd like me to prepare."),
        ("Gentle nudge on the pending item",
         "Just a gentle nudge on the item we discussed — whenever you have "
         "bandwidth."),
        ("Did you get a chance to review my notes?",
         "Wondering if you had a chance to look at the notes I shared. "
         "Happy to walk through them together."),
        ("Following up on our call yesterday",
         "Following up on a couple of points from our call yesterday — let "
         "me know your thoughts when you can."),
        ("Any update on the timeline we discussed?",
         "Just checking if there's any update on the timeline we talked "
         "about last time."),
        ("Wanted to follow up before the week ends",
         "Following up before the week wraps up — let me know if this is "
         "still on your radar."),
        ("Checking if you received my last message",
         "Just making sure my last message reached you — happy to resend "
         "if needed."),
        ("Following up on the introduction you offered",
         "Following up on the introduction you mentioned — would still "
         "love to connect if it's convenient for you."),
        ("A quick follow-up on next steps",
         "Wanted to follow up quickly on what the next steps look like "
         "from here."),
        ("Circling back after the holidays",
         "Circling back now that things have settled after the holidays — "
         "any update on this?"),
        ("Following up on the feedback form",
         "Just following up on the feedback form I sent over — let me know "
         "if you have any questions."),
        ("Checking in on our earlier discussion",
         "Wanted to check in on our earlier discussion and see where "
         "things stand."),
        ("Any progress on this from your side?",
         "Curious if there's been any progress on this from your side "
         "whenever you have a chance to update me."),
        ("Following up ahead of the deadline",
         "Following up ahead of the deadline just to confirm we're still "
         "on track."),
        ("Wanted to reconnect on this topic",
         "Wanted to reconnect on this topic since it's been a little while "
         "— happy to pick it back up whenever works."),
        ("A friendly follow-up on the document",
         "A friendly follow-up on the document I shared — let me know if "
         "it needs any changes."),
    ],
    "News & Promotions": [
        ("This week in campus news",
         "Catch up on what happened around campus this week, from club "
         "events to the latest research highlights from the department."),
        ("New arrivals: fresh drops just landed",
         "Check out the newest additions to our store this week, plus a "
         "limited-time discount for students."),
        ("Monthly newsletter: alumni spotlight and upcoming events",
         "In this month's newsletter: an alumni spotlight, a roundup of "
         "upcoming department events, and a summary of recent publications."),
        ("You're invited: guest lecture series kicks off next month",
         "We're excited to announce our new guest lecture series featuring "
         "speakers from across the industry. Mark your calendar."),
        ("50% off your next order — this weekend only",
         "Enjoy half off your next purchase this weekend only. Shop the sale "
         "before it ends."),
        ("Meet the new student council for this year",
         "Get to know the newly elected student council members and what "
         "they're planning for the semester ahead."),
        ("Product update: here's what's new this quarter",
         "We've shipped a number of improvements this quarter. Here's a "
         "quick look at what's new and what's coming next."),
        ("Sports club recap: highlights from last month's matches",
         "A quick recap of last month's matches, including standout "
         "performances and the updated league table."),
        ("Your weekly digest is here",
         "Here's your weekly digest with the top stories and updates you "
         "might have missed."),
        ("Announcing our new mobile app",
         "We're thrilled to announce the launch of our new mobile app, "
         "packed with features requested by our community."),
        ("This month's event calendar is here",
         "Take a look at everything happening on campus this month, all "
         "in one place."),
        ("Flash sale: 24 hours only",
         "Our flash sale is live for the next 24 hours only. Don't miss "
         "these limited-time prices."),
        ("Webinar recording now available",
         "Missed our last webinar? The full recording is now available to "
         "watch on demand."),
        ("Community spotlight: member of the month",
         "Meet this month's featured community member and learn about "
         "their journey."),
        ("Your subscription benefits just got better",
         "We've added new perks to your subscription — take a look at "
         "what's new."),
        ("Save the date: annual fest is back",
         "Mark your calendars — the annual fest returns next month with a "
         "full lineup of events."),
        ("New blog post: tips for the semester ahead",
         "Check out our latest blog post with tips to make the most of "
         "the semester ahead."),
        ("Early bird pricing ends soon",
         "Early bird pricing for the upcoming conference ends this week — "
         "register to lock in the discount."),
        ("Your loyalty points are about to expire",
         "A reminder that your accumulated loyalty points are set to "
         "expire at the end of the month."),
        ("Introducing our new referral program",
         "We've launched a new referral program — invite friends and earn "
         "rewards together."),
        ("Recap: highlights from last week's town hall",
         "Catch up on the key highlights and announcements from last "
         "week's town hall."),
        ("Season sale starts this Friday",
         "Our biggest seasonal sale of the year starts this Friday. Get "
         "ready for great deals."),
        ("New podcast episode: student stories",
         "Our latest podcast episode features stories from students "
         "across different departments."),
        ("Your monthly account summary is ready",
         "Your monthly account summary and activity overview is now ready "
         "to view."),
        ("Upcoming maintenance window this weekend",
         "We'll be performing scheduled maintenance this weekend; some "
         "services may be briefly unavailable."),
        ("New feature rollout: what's changed",
         "We've rolled out a new set of features this week. Here's a "
         "quick overview of what's changed."),
        ("Join our reader survey and win prizes",
         "Share your feedback in our quick reader survey for a chance to "
         "win a prize."),
        ("Highlights from this year's alumni meet",
         "A quick roundup of highlights and photos from this year's "
         "alumni meet-up."),
        ("Limited edition merchandise now live",
         "Our limited edition merchandise collection just went live — "
         "while supplies last."),
        ("Your weekly roundup of top discussions",
         "Here's a roundup of the most active discussions from the "
         "community this week."),
    ],
    "Spam": [
        ("You have won a prize! Claim now",
         "Congratulations! You have been selected to win a cash prize. "
         "Click the link below and enter your bank details to claim it."),
        ("URGENT: verify your account or it will be suspended",
         "We detected unusual activity on your account. Verify your login "
         "credentials immediately by clicking this link to avoid suspension."),
        ("Get rich quick with this one simple trick",
         "Make thousands of dollars a week from home with this simple "
         "trick. No experience needed, guaranteed results."),
        ("Your package could not be delivered",
         "Your package delivery failed due to an incomplete address. Click "
         "here and pay a small redelivery fee to reschedule."),
        ("Cheap medications, no prescription needed",
         "Buy discounted medications online without a prescription. Fast, "
         "discreet shipping worldwide."),
        ("Hot singles in your area want to meet you",
         "Someone near you is interested in chatting. Sign up now to see "
         "who's waiting to meet you."),
        ("Free iPhone giveaway — limited slots left",
         "You've been randomly selected for a free iPhone giveaway. Only a "
         "few slots remain, claim yours before it's too late."),
        ("Your loan has been pre-approved",
         "You are pre-approved for an instant loan with no credit check. "
         "Apply now and get funds within minutes."),
        ("Increase your followers instantly",
         "Buy followers and likes instantly at unbeatable prices. Boost "
         "your social presence today."),
        ("Security alert: unusual sign-in attempt detected",
         "An unusual sign-in attempt was detected from a new device. If "
         "this wasn't you, click here to secure your account immediately."),
        ("Claim your inheritance now",
         "A distant relative has left you an inheritance. Provide your "
         "bank details to begin the transfer process."),
        ("You are the lucky winner of our lottery",
         "Your email address has won our international lottery draw. "
         "Reply with your details to claim your winnings."),
        ("Work from home and earn $5000 a week",
         "Earn $5000 a week working from home with no experience "
         "required. Sign up with your payment details today."),
        ("Your antivirus subscription has expired — renew now",
         "Your antivirus protection has expired, leaving your device at "
         "risk. Click here to renew immediately with your card details."),
        ("Congratulations, you qualify for a free cruise",
         "You've been selected for a free luxury cruise. Confirm your "
         "spot by entering your credit card for a small booking fee."),
        ("Double your bitcoin investment in 24 hours",
         "Invest in our exclusive program and double your bitcoin in just "
         "24 hours. Limited spots available."),
        ("Your email storage is full — click to upgrade",
         "Your mailbox storage is full. Click this link and log in with "
         "your credentials to upgrade immediately."),
        ("Weight loss secret doctors don't want you to know",
         "Lose 10 kilos in a week with this secret formula doctors don't "
         "want you to know about."),
        ("You've been selected for a mystery shopper program",
         "You qualify for our mystery shopper program. Send your banking "
         "details to receive your first assignment."),
        ("Act now: your domain is about to expire",
         "Your domain registration is about to expire. Renew immediately "
         "by providing your payment information."),
        ("Exclusive investment opportunity — guaranteed returns",
         "Join our exclusive investment club for guaranteed high returns "
         "with zero risk."),
        ("Your computer has a virus — call this number now",
         "Our scan detected a virus on your computer. Call the number "
         "below immediately for remote support."),
        ("Free trial — no credit card needed (terms apply)",
         "Start your free trial today, no credit card needed — just "
         "confirm your account details to begin."),
        ("Unclaimed funds are waiting for you",
         "Records show unclaimed funds under your name. Submit your "
         "personal details to process the claim."),
        ("You've been pre-selected for a platinum credit card",
         "You are pre-selected for our platinum credit card with no "
         "annual fee. Apply now with your SSN and income details."),
        ("Your subscription will auto-renew — cancel here",
         "Your subscription is about to auto-renew. Click here and enter "
         "your card details to cancel."),
        ("Earn a degree in 5 days — no exams required",
         "Get a recognized degree in just 5 days, no exams or coursework "
         "required."),
        ("You have a new voicemail — listen now",
         "You have a new voicemail message waiting. Click the link and "
         "sign in to listen now."),
        ("Refund pending — confirm your bank details",
         "A refund is pending on your account. Confirm your bank details "
         "to receive the payment."),
        ("Your parcel is held at customs — pay to release",
         "Your parcel is being held at customs. Pay the release fee below "
         "to have it delivered."),
    ],
}


def main():
    rows = []
    for category, pairs in SAMPLES.items():
        for subject, body in pairs:
            rows.append({"subject": subject, "body": body, "category": category})

    random.shuffle(rows)

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "body", "category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} labeled examples to {OUT_PATH}")
    counts = {cat: len(pairs) for cat, pairs in SAMPLES.items()}
    for cat, n in counts.items():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()