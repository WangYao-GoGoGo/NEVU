<View>
            <HyperText name="hyperlink_text" value="&lt;a href='https://ii.slis.tsukuba.ac.jp/echv_event_system/subevent_details.html?uuid=$uuid&amp;phase_version=$phase_version&amp;annotation_filename=$annotation_filename' target='_blank'&gt;📋 Go to Detail Page!&lt;/a&gt;" />
            <Header value="*Annotation Question:" size="6"/>
            <Text name="question" value="$questions"/>
            <Header value="*Explanation from Large Language Models:" size="6"/>
            <Text name="explanation" value="$explanation"/>
            <Header value="*Choose one answer(required):" size="6"/>
            <Choices name="subevent_hv_verification" toName="question" choice="single" showInLine="true">
                <Choice value="Yes"/>
                <Choice value="No"/>
                <Choice value="Unknown"/>
            </Choices>

            <Header value="*Reason for your choice (required):"/>
            <Choices name="reason_choice" toName="question" choice="multiple" showInLine="false">
                <Header value="=========If you choose Yes========" size="6" />
                <Choice value="Actor's behavior/statement directly reflects the value"/>
                <Choice value="Actor's behavior/statement indirectly reflects the value"/>
                <Header value="=========If you choose No or Unknown========"  size="6"/>
                <Choice value="Actor's behavior/statement does not involve the value"/>
                <Choice value="Actor's behavior/statement expresses the opposite value"/>
                <Choice value="The subevent is irrelevant to the value"/>
                <Choice value="Insufficient contextual information in the subevent"/>
                <Choice value="Actor reference is unclear (ambiguous/pronoun issue)"/>
                <Choice value="The value expression is vague or unclear"/>
                <Choice value="Other"/>
            </Choices>

            <Header value="If Other, please add explanation (optional):" size="6" />
            <TextArea name="reason_free_text" toName="question"
                      placeholder="Write your reason here (optional)"
                      rows="3" maxSubmissions="1"/>

            <Header value="(if you choose yes) Does this human value significantly contribute to the overall meaning or stance of the article? (required)"/>
            <Choices name="importance_choice" toName="question" choice="single" showInLine="false">
                <Choice value="Yes" hint="Removing this value would clearly change the meaning or stance of the article.">Yes</Choice>
                <Choice value="Partial" hint="Removing this value would cause minor or indirect change, but the main meaning remains.">Partial</Choice>
                <Choice value="No" hint="Removing this value would not affect the overall meaning or stance.">No</Choice>
            </Choices>

            <Header value="====================Additional information============================" />
            <Header value="uuid:" size="6" />
            <Text name="uuid" value="$uuid"/>
            <Header value="guid:" size="6" />
            <Text name="guid" value="$guid"/>
            <Header value="value_direction:" size="6" />
            <Text name="value_direction" value="$value_direction"/>
            <Header value="human_value_type:" size="6" />
            <Text name="human_value_type" value="$human_value_type"/>
            <Header value="Voting count:" size="6" />
            <Text name="voting_count" value="$voting_num"/>
        </View>