-- comp9311 22T2 Project 1

-- Q1:
create or replace view Q1(subject_code)
as
select code from subjects
where uoc>12 and career='PG' and offeredby=(select id from orgunits where longname like'%School of Computer Science and Engineering%')
--... SQL statements, possibly using other views/functions defined by you ...
;

-- Q2:
create or replace view Q2(course_id)
as
select courses.id from courses
where semester = (select id from semesters where year='2002' and term ='S1') and courses.id in (select course from classes group by course having count(distinct ctype)>=3);
--... SQL statements, possibly using other views/functions defined by you ...
;

-- Q3:
create or replace view Q3_1(student,semester)
as
select a.id,c.semester from students a,course_enrolments b,courses c
where a.id=b.student and a.stype='local' and b.course=c.id and c.subject in (select id from subjects where code='COMP9311');

create or replace view Q3_2(student,semester)
as
select a.id,c.semester from students a,course_enrolments b,courses c 
where a.id=b.student and a.stype='local' and b.course=c.id and c.subject in (select id from subjects where code='COMP9331');

create or replace view Q3_3(student)
as
select distinct a.student from Q3_1 a,Q3_2 b 
where a.student = b.student and  a.semester = b.semester;

create or replace view Q3_4(student)
as
select distinct a.student from Q3_3 a,students b 
where a.student=b.id and stype='local';

create or replace view Q3(unsw_id, name)
as
select a.unswid, a.name from people a,Q3_4 b 
where a.id=b.student
--... SQL statements, possibly using other views/functions defined by you ...
;

-- Q4:
create or replace view Q4_1(course_id, subject_code,semester)
as
select courses.id, subject, semester from courses where subject =(select subjects.id from subjects where code='COMP9311') and courses.semester in (select semesters.id from semesters where year between 2009 and 2012)
;
create or replace view Q4_2(semester_id,term_name, year)
as
select id,name,year from semesters where year between 2009 and 2012
;
create or replace view Q4_3(student,course_id, mark)
as
select student,course,mark from course_enrolments where course in (select course_id from q4_1) and mark is not null
;
create or replace view Q4_4(course_id,count_lt50)
as
select course_id, count(student) from Q4_3 where mark<50 group by course_id
;
create or replace view Q4_5(course_id, total)
as
select course_id, count(student) from q4_3 group by course_id
;
create or replace view Q4_6(course_id,count_lt50,total)
as
select a.course_id,cast(count_lt50 as numeric(10,6)),cast(total as numeric(10,6)) from q4_4 a,q4_5 b where a.course_id = b.course_id;
;
create or replace view Q4_7(course_id, max_fail_rate)
as
select course_id, count_lt50/total as max_fail_rate from q4_6 where count_lt50/total = (select max(count_lt50/total) from q4_6)
;
create or replace view Q4_8(course_id,term_name)
as
select course_id, term_name from q4_1,q4_2 where semester=semester_id
;

create or replace view Q4(term, max_fail_rate)
as
select term_name, cast(max_fail_rate as numeric(10,4)) from q4_7 a,q4_8 b where a.course_id=b.course_id
--... SQL statements, possibly using other views/functions defined by you ...
;

-- Q5:
create or replace view Q5_1(student,course,grade)
as
select student,course,grade from course_enrolments, students where student = id and stype='intl'
;
create or replace view Q5_2(student,course_cnt,HD_cnt)
as
select student,count(course) as course_cnt, sum(case grade when 'HD' then 1 else 0 end) as HD_cnt from q5_1  group by student
;
create or replace view Q5_3(student,course_cnt,HD_cnt)
as
select student,course_cnt,HD_cnt from q5_2 where course_cnt=HD_cnt
;
create or replace view Q5(unsw_id, student_name)
as
select unswid, name from people where id in (select student from q5_3)
--... SQL statements, possibly using other views/functions defined by you ...
;

-- Q6:
create or replace view Q6_1(id,utype,name,longname,unswid)
as
select id,utype,name,longname,unswid from orgunits where utype = (select id from orgunit_types where name='School')
;
create or replace view Q6_2(id,code,name,offeredby,stype)
as
select id,code,name,offeredby,stype from streams where offeredby in (select id from q6_1)
;
create or replace view Q6_3(school_id,stream_count)
as
select offeredby,count(q6_2.id) from q6_2 group by offeredby
;
create or replace view Q6_4(school_id,stream_count)
as
select offeredby,count(q6_2.id) from q6_2 where offeredby = (select id from q6_1 where longname='School of Chemical Engineering') group by offeredby
;
create or replace view Q6(school_name, stream_count)
as
select c.longname,a.stream_count from q6_3 a, q6_4 b,orgunits c where a.stream_count > b.stream_count and a.school_id = c.id
--... SQL statements, possibly using other views/functions defined by you ...
;

-- Q7:
create or replace view Q7_1(class_id,course,room)
as
select id,course,room from classes where course in (select id from courses where semester = (select id from semesters where year = '2010' and term = 'S2'))
;
create or replace view Q7_2(class_id,course,room,building)
as
select class_id, course,a.room,building from q7_1 a, rooms where a.room = rooms.id
;
create or replace view Q7_3(course,building_cnt)
as
select course, count(distinct building) as buliding_cnt from q7_2 group by course having count(distinct building) > 3
;
create or replace view Q7_4(course)
as
select course from q7_3
where course in (select courses.id from courses where subject in (select id from subjects where offeredby=(select id from orgunits where longname like'%School of Computer Science and Engineering%')))
;

create or replace view Q7(course_id, staff_name)
as
select course,name from course_staff, people where course in (select course from q7_4) and staff=id
--... SQL statements, possibly using other views/functions defined by you ...
;

-- Q8:
create or replace view Q8_1(program_enrol_id,student,semester)
as
select id,student,semester,program from program_enrolments where program in (select program from program_degrees where abbrev='MSc')
--... condition 1
;
create or replace view Q8_2(student,course,mark,grade)
as
select a.student,course,mark,grade from course_enrolments a where course in (select id from courses where semester = (select id from semesters where year='2012' and term ='S2'))
and exists (select b.semester from q8_1 b where b.semester = (select id from semesters where year='2012' and term ='S2') and b.student=a.student)
;
create or replace view Q8_3(student)
as
select student from (select student,count(course) as course_cnt, sum(case when mark>=50 then 1 else 0 end) as pass_cnt from q8_2 group by student) as a
where pass_cnt>=1
;
--... condition 1 and 2
create or replace view Q8_4(student,course,mark,grade)
as
select a.student,course,mark,grade from course_enrolments a where course in (select id from courses where semester in (select id from semesters where year<2013))
and exists (select b.semester from q8_1 b where b.semester in (select id from semesters where year<2013) and b.student=a.student)
and student in (select student from q8_3)
;
create or replace view Q8_5(student,average)
as
select student, avg(mark) from q8_4 group by student having avg(mark)>=80
;
--... condition 1, 2 and 3
create or replace view Q8_6(student,course,mark,grade)
as
select student,course,mark,grade from q8_4 where student in (select student from q8_5)
;
create or replace view Q8_7(student)
as
select student,subject,semester,uoc from q8_6 a join (select id,subject, semester from courses) as b on b.id=a.course join (select id,uoc from subjects) as c on c.id=b.subject
;
-- create or replace view Q8_8(student)
-- as
-- select student,subject,a.semester,program from q8_7 a, (select semester, program from q8_1) as b 
-- where b.semester = a.semester and 
-- ;
create or replace view Q8(unsw_id, name)
as
select unswid, name from people where id in (select student from q8_5)
--... SQL statements, possibly using other views/functions defined by you ...
;

-- Q9:
create or replace view q9_1(student,num1)
as
select distinct a.student,count(distinct a.course) from course_enrolments a,courses b,semesters c where a.mark>=0 and a.course=b.id and b.semester=c.id and c.year='2012' and c.term='S1' group by a.student;

create or replace view Q9_2(student,num2)
as
select distinct a.student,count(distinct a.course) from course_enrolments a,courses b,semesters c where a.mark>=50 and a.course=b.id and b.semester=c.id and c.year='2012' and c.term='S1' group by a.student;

create or replace view Q9_3(student,num1,num2)
as
select distinct a.student,a.num1,b.num2 from Q9_1 a left join Q9_2 b on a.student=b.student;

create or replace view Q9_31(student,num1,num2)
as
select distinct student,num1,coalesce(num2, 0) from Q9_3;

create or replace view Q9_4(student,rate,num1)
as
select distinct a.student,a.num2::real/num1,a.num1 from Q9_31 a;

create or replace view Q9_41(unswid,name,rate,num1)
as
select a.unswid,a.name,b.rate,b.num1 from people a,Q9_4 b where a.id=b.student and a.unswid::text like '313%';


create or replace view Q9_5(unswid,name)
as
select unswid,name from Q9_41 where rate=0 and num1>1;

create or replace view Q9_6(unswid,name)
as
select unswid,name from Q9_41 where rate<=0.5 and num1>1;

create or replace view Q9_7(unswid,name)
as
select unswid,name from Q9_41 where rate>0.5 and num1>1;

create or replace view Q9_8(unswid,name)
as
select unswid,name from Q9_41 where rate=0 and num1=1;

create or replace view Q9_9(unswid,name)
as
select unswid,name from Q9_41 where rate=1 and num1=1;

create or replace view Q9(unswid,name,academic_standing)
as
select a.unswid,a.name,(case when a.unswid in (select unswid from Q9_5) then  'Probation' when a.unswid in (select unswid from Q9_6) then 'Referral' when a.unswid in (select unswid from Q9_8) then 'Referral' else 'Good' end) from people a,Q9_41 b where a.unswid=b.unswid order by a.unswid
--... SQL statements, possibly using other views/functions defined by you ...
;

-- Q10

create or replace function 
	Q10(staff_id integer) returns setof text
as $$
declare
	r record;
begin
	for r in select * from affiliations join (select id, longname from orgunits) as a on a.id=orgunit join (select id,name from staff_roles) as b on b.id=role where staff= staff_id order by starting,b.name,a.longname
	loop
		return next r.longname||'/'||r.name||'/'||r.starting;
	end loop;
	return;
end;
--... SQL statements, possibly using other views/functions defined by you ...
$$ language plpgsql;

-- Q11
create or replace function 
	Q11(year courseyeartype, term character(2), orgunit_id integer) returns setof text
as $$
--... SQL statements, possibly using other views/functions defined by you ...
$$ language plpgsql;

-- Q12
create or replace function 
	Q12(code character(8)) returns setof text
as $$
declare
	r record;	
	_code varchar = upper(code);
begin
		for r in select id,a.code,_prereq from subjects as a where substring(a.code,1,4)=substring(_code,1,4) and _prereq like '%'||_code||'%'
	loop
		return next r.code;
	end loop;
	return;
end;
--... SQL statements, possibly using other views/functions defined by you ...
$$ language plpgsql;