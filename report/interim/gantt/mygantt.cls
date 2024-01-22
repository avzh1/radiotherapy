\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mygantt}[My Gantt Chart]
\LoadClass[margin=5pt,12pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage{url}
\usepackage{tikz}
\usetikzlibrary{backgrounds,calc,calendar}
\usepackage{pgfgantt}
\usepackage{pgfcalendar}
\usepackage{xcolor}
\usepackage{etoolbox}
\usepackage{calc}
\usepackage{ifthen}
\usepackage[nomessages]{fp}
\usepackage{fmtcount}
\usepackage{pgffor}

\def\myxunit{.5cm}%width of 1 day
\def\myyunittitle{.5cm}%height of 1 title line
\def\myyunitchart{1cm}%height of 1 chart line

\renewcommand{\familydefault}{\sfdefault}

%set colours
\definecolor{barblue}{RGB}{86,196,197}
\definecolor{barprogress}{RGB}{245,141,116}
\definecolor{linkred}{RGB}{228,109,86}


%set link types
\setganttlinklabel{s-s}{START-TO-START}
\setganttlinklabel{f-s}{FINISH-TO-START}
\setganttlinklabel{f-f}{FINISH-TO-FINISH}

\def\pgfcalendarweekdayletter#1{% define the name of weekdays + formatting
    \ifcase#1M\or T\or W\or T\or F\or S\or \textcolor{red}{\textbf{S}}\fi
}   

\ganttset{%
    calendar week text={%
    \pgfcalendarmonthshortname{\startmonth}~\startday, \startyear%
    }%
}

% %commands for progress reporting
% \newcommand{\one}   {   999  } 
% \newcommand{\two}   {   999  } 
% \newcommand{\three} {   999  } 
% \newcommand{\four}  {   999  } 
% \newcommand{\five}  {   999  } 
% \newcommand{\six}   {   999  }

\newcommand{\getcurrentref}[1]{%
  \ifnum\value{#1}=0 ??\else\csname the#1\endcsname\fi
}


\setcounter{section}{0}
\setcounter{subsection}{1}

\newcommand{\s}{\thesection{} }
\newcommand{\su}{\thesubsection{\addtocounter{subsection}{1}}}

\newcommand{\drawline}[3]{\ganttvrule[vrule offset=.5, vrule/.append style={draw=#2!80,dash pattern=on 3.0pt off 4.5pt,line width=1.5pt}]{\bfseries\small{#1}}{#3}}